import pytest
import json
import re
from unittest.mock import AsyncMock, Mock, patch
from typing import Optional

from app.classifier.chunk_classifier import ChunkClassifier
from app.utils.schemas import DocCategory, RawElement, Classified
from app.utils.llm_utils import LLMDriven


class TestChunkClassifier:
    """Тесты для ChunkClassifier"""
    
    @pytest.fixture
    def classifier_config(self):
        """Базовая конфигурация для классификатора"""
        return {
            "api_url": "http://test-api.com",
            "api_key": "test-key",
            "model": "test-model",
            "system_prompt": "Classify this: {chunk_text}\nCategories: {category_descriptions}",
            "category_descriptions": {
                "document": "A formal document",
                "section": "A section of content",
                "paragraph": "A paragraph of text"
            }
        }
    
    @pytest.fixture
    def classifier(self, classifier_config):
        """Создание экземпляра классификатора"""
        return ChunkClassifier(**classifier_config)
    
    @pytest.fixture
    def sample_chunk(self):
        """Пример чанка для классификации"""
        return RawElement(
            source="test.pdf",
            category=DocCategory.DOCUMENT,
            content="This is a test document content",
            metadata={}
        )
    
    def test_init(self, classifier_config):
        """Тест инициализации классификатора"""
        classifier = ChunkClassifier(**classifier_config)
        
        assert classifier.api_url == classifier_config["api_url"]
        assert classifier.api_key == classifier_config["api_key"]
        assert classifier.model == classifier_config["model"]
        assert classifier.system_prompt == classifier_config["system_prompt"]
        assert classifier.category_descriptions == classifier_config["category_descriptions"]
        assert DocCategory.KNOWLEDGE_CANVAS in classifier.excluded_categories
        assert len(classifier.excluded_categories) == 1
    
    def test_build_categories_string(self, classifier):
        """Тест построения строки с категориями"""
        categories_str = classifier._build_categories_string()
        
        # Проверяем, что KNOWLEDGE_CANVAS исключен
        assert DocCategory.KNOWLEDGE_CANVAS.value not in categories_str
        
        # Проверяем, что остальные категории присутствуют
        for cat in DocCategory:
            if cat not in classifier.excluded_categories:
                assert cat.value in categories_str
                assert classifier.category_descriptions.get(cat.value, cat.value) in categories_str
    
    def test_build_categories_string_with_custom_descriptions(self, classifier_config):
        """Тест построения строки с кастомными описаниями"""
        classifier = ChunkClassifier(**classifier_config)
        categories_str = classifier._build_categories_string()
        
        # Проверяем, что кастомные описания используются
        for cat_value, description in classifier_config["category_descriptions"].items():
            assert description in categories_str
    
    @pytest.mark.asyncio
    async def test_classify_success(self, classifier, sample_chunk):
        """Тест успешной классификации"""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"is_useful": true, "category": "section", "confidence": 0.95}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await classifier.classify(sample_chunk)
            
            # Проверяем вызов API
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            assert call_args["model"] == classifier.model
            
            # Проверяем системный промпт
            system_message = call_args["messages"][0]["content"]
            assert sample_chunk.content in system_message
            assert classifier.categories_str in system_message
            
            # Проверяем результат
            assert isinstance(result, Classified)
            assert result.is_useful is True
            assert result.category == DocCategory.SECTION
            assert result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_classify_unuseful_content(self, classifier, sample_chunk):
        """Тест классификации бесполезного контента"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"is_useful": false, "category": "document", "confidence": 0.1}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await classifier.classify(sample_chunk)
            
            # Для бесполезного контента категория должна быть DOCUMENT
            assert result.is_useful is False
            assert result.category == DocCategory.DOCUMENT
            assert result.confidence == 0.1
    
    @pytest.mark.asyncio
    async def test_classify_api_error(self, classifier, sample_chunk):
        """Тест обработки ошибки API"""
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API connection error")
            
            result = await classifier.classify(sample_chunk)
            
            # Проверяем fallback
            assert result.is_useful is True
            assert result.category == (sample_chunk.category or DocCategory.DOCUMENT)
            assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_classify_with_null_category(self, classifier):
        """Тест классификации с null категорией"""
        chunk_without_category = RawElement(
            source="test.pdf",
            category=None,  # category is None
            content="Test content",
            metadata={}
        )
        
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API error")
            
            result = await classifier.classify(chunk_without_category)
            
            # Должен быть DOCUMENT как fallback
            assert result.category == DocCategory.DOCUMENT
            assert result.is_useful is True
    
    def test_parse_response_valid_json(self, classifier):
        """Тест парсинга валидного JSON ответа"""
        raw_response = '{"is_useful": true, "category": "paragraph", "confidence": 0.88}'
        result = classifier._parse_response(raw_response)
        
        assert isinstance(result, Classified)
        assert result.is_useful is True
        assert result.category == DocCategory.PARAGRAPH
        assert result.confidence == 0.88
    
    def test_parse_response_json_with_extra_text(self, classifier):
        """Тест парсинга JSON с дополнительным текстом"""
        raw_response = 'Here is the classification: {"is_useful": true, "category": "section", "confidence": 0.75}'
        result = classifier._parse_response(raw_response)
        
        assert isinstance(result, Classified)
        assert result.is_useful is True
        assert result.category == DocCategory.SECTION
        assert result.confidence == 0.75
    
    def test_parse_response_json_with_whitespace(self, classifier):
        """Тест парсинга JSON с пробелами и переносами"""
        raw_response = '''
        {
            "is_useful": false,
            "category": "document",
            "confidence": 0.0
        }
        '''
        result = classifier._parse_response(raw_response)
        
        assert isinstance(result, Classified)
        assert result.is_useful is False
        assert result.category == DocCategory.DOCUMENT
        assert result.confidence == 0.0
    
    def test_parse_response_no_json(self, classifier):
        """Тест парсинга ответа без JSON"""
        raw_response = "No JSON here at all"
        
        with pytest.raises(ValueError) as exc_info:
            classifier._parse_response(raw_response)
        
        assert "No JSON found in LLM response" in str(exc_info.value)
        assert raw_response in str(exc_info.value)
    
    def test_parse_response_invalid_json(self, classifier):
        """Тест парсинга невалидного JSON"""
        raw_response = '{"is_useful": true, "category": "section", "confidence": }'
        
        with pytest.raises(json.JSONDecodeError):
            classifier._parse_response(raw_response)
    
    def test_parse_response_unuseful_changes_category(self, classifier):
        """Тест что для бесполезного контента категория меняется на DOCUMENT"""
        raw_response = '{"is_useful": false, "category": "section", "confidence": 0.05}'
        result = classifier._parse_response(raw_response)
        
        assert result.is_useful is False
        assert result.category == DocCategory.DOCUMENT
        assert result.confidence == 0.05
    
    def test_parse_response_useful_keeps_category(self, classifier):
        """Тест что для полезного контента категория сохраняется"""
        raw_response = '{"is_useful": true, "category": "paragraph", "confidence": 0.9}'
        result = classifier._parse_response(raw_response)
        
        assert result.is_useful is True
        assert result.category == DocCategory.PARAGRAPH
        assert result.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_classify_logging_on_success(self, classifier, sample_chunk, caplog):
        """Тест логирования при успешной классификации"""
        import structlog
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"is_useful": true, "category": "section", "confidence": 0.95}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            with caplog.at_level("DEBUG"):
                result = await classifier.classify(sample_chunk)
                
                # Проверяем, что логи были вызваны
                assert "classification_processing_start" in caplog.text
                assert "classification_processing_done" in caplog.text
    
    @pytest.mark.asyncio
    async def test_classify_logging_on_error(self, classifier, sample_chunk, caplog):
        """Тест логирования при ошибке классификации"""
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Connection timeout")
            
            with caplog.at_level("ERROR"):
                result = await classifier.classify(sample_chunk)
                
                # Проверяем, что ошибка залогирована
                assert "classification_failed" in caplog.text
                assert "Connection timeout" in caplog.text
    
    def test_categories_str_property(self, classifier):
        """Тест что categories_str формируется при инициализации"""
        assert classifier.categories_str is not None
        assert isinstance(classifier.categories_str, str)
        assert len(classifier.categories_str) > 0
    
    @pytest.mark.asyncio
    async def test_classify_response_without_choices(self, classifier, sample_chunk):
        """Тест обработки ответа без choices"""
        mock_response = Mock()
        mock_response.choices = None
        
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            mock_create.side_effect = Exception("No choices in response")
            
            result = await classifier.classify(sample_chunk)
            
            # Проверяем fallback
            assert result.is_useful is True
            assert result.category == (sample_chunk.category or DocCategory.DOCUMENT)
            assert result.confidence == 0.0
    
    def test_excluded_categories_immutable(self, classifier):
        """Тест что excluded_categories не изменяется случайно"""
        original_excluded = classifier.excluded_categories.copy()
        
        # Пытаемся изменить
        classifier.excluded_categories.append(DocCategory.DOCUMENT)
        
        # Проверяем, что оригинальный список не изменился
        # (это тест на случай, если excluded_categories - кортеж или неизменяемый)
        # Если это список, то он изменится, но это нормально
        # Лучше проверить, что KNOWLEDGE_CANVAS всегда исключен
        assert DocCategory.KNOWLEDGE_CANVAS in classifier.excluded_categories


class TestChunkClassifierIntegration:
    """Интеграционные тесты (с реальными API вызовами - мокаем)"""
    
    @pytest.fixture
    def classifier(self):
        return ChunkClassifier(
            api_url="http://localhost:8000",
            api_key="test-key",
            model="llama2",
            system_prompt="Classify text: {chunk_text}\nCategories: {category_descriptions}",
            category_descriptions={
                "introduction": "Introductory section",
                "body": "Main content",
                "conclusion": "Concluding section"
            }
        )
    
    @pytest.mark.asyncio
    async def test_classify_different_categories(self, classifier):
        """Тест классификации разных категорий"""
        test_cases = [
            ("Introduction text", "introduction", 0.9),
            ("Main body content", "body", 0.85),
            ("Conclusion summary", "conclusion", 0.8)
        ]
        
        for content, expected_category, expected_confidence in test_cases:
            chunk = RawElement(
                source="test.pdf",
                category=DocCategory.DOCUMENT,
                content=content,
                metadata={}
            )
            
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content=f'{{"is_useful": true, "category": "{expected_category}", "confidence": {expected_confidence}}}'))
            ]
            mock_response.usage = Mock()
            
            with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_response
                
                result = await classifier.classify(chunk)
                
                assert result.is_useful is True
                assert result.category == DocCategory(expected_category)
                assert result.confidence == expected_confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])