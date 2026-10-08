import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from know_everything_ai.classifier.chunk_classifier import ChunkClassifier
from know_everything_ai.prompts import CHUNK_CLASSIFIER_PROMPT
from know_everything_ai.schemas import Classified, DocCategory, RawElement

#: Long enough to clear CLASSIFIER_TRASH_SIZE, so the tests exercise the model
#: call rather than the cheap length gate (which has its own test below).
SAMPLE_CONTENT = (
    "Настоящий регламент определяет порядок обработки входящих заявок "
    "и устанавливает сроки рассмотрения для всех категорий обращений. "
    "Заявка подлежит регистрации в системе не позднее одного рабочего дня."
)


def _completion(content: str) -> Mock:
    response = Mock()
    response.choices = [Mock(message=Mock(content=content))]
    response.usage = Mock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    return response


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
            },
            # Disable the length gate so model behaviour is tested in isolation.
            "trash_size": 0,
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
            content=SAMPLE_CONTENT,
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
        assert classifier.trash_size == 0

    def test_build_categories_string(self, classifier):
        """Тест построения строки с категориями"""
        categories_str = classifier._build_categories_string()

        # Проверяем, что KNOWLEDGE_CANVAS исключен
        assert DocCategory.KNOWLEDGE_CANVAS.value not in categories_str

        # Проверяем, что остальные категории присутствуют
        for cat in DocCategory:
            if cat not in classifier.excluded_categories:
                assert cat.value in categories_str

    def test_build_categories_string_with_custom_descriptions(self, classifier_config):
        """Тест построения строки с кастомными описаниями"""
        classifier = ChunkClassifier(**classifier_config)
        categories_str = classifier._build_categories_string()

        # Проверяем, что кастомные описания используются
        for _cat_value, description in classifier_config["category_descriptions"].items():
            assert description in categories_str

    @pytest.mark.asyncio
    async def test_classify_success(self, classifier, sample_chunk):
        """Тест успешной классификации"""
        mock_response = _completion(
            '{"is_useful": true, "category": "section", "confidence": 0.95}'
        )

        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await classifier.classify(sample_chunk)

            # Проверяем вызов API
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            assert call_args["model"] == classifier.model

            # Проверяем раскладку по ролям: провайдеру нужен и system, и user.
            messages = call_args["messages"]
            assert [m["role"] for m in messages] == ["system", "user"]
            assert sample_chunk.content in messages[1]["content"]
            assert classifier.categories_str in messages[1]["content"]
            assert sample_chunk.content not in messages[0]["content"]

            # Проверяем результат
            assert isinstance(result, Classified)
            assert result.is_useful is True
            assert result.category == DocCategory.DOCUMENT
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_classify_unuseful_content(self, classifier, sample_chunk):
        """Тест классификации бесполезного контента"""
        mock_response = _completion(
            '{"is_useful": false, "category": "document", "confidence": 0.1}'
        )

        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await classifier.classify(sample_chunk)

            # Для бесполезного контента категория должна быть DOCUMENT
            assert result.is_useful is False
            assert result.category == DocCategory.DOCUMENT
            assert result.confidence == 0.1

    @pytest.mark.asyncio
    async def test_classify_api_error_drops_chunk(self, classifier, sample_chunk):
        """Ошибка API отбрасывает фрагмент, а не пропускает его в базу."""
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API connection error")

            result = await classifier.classify(sample_chunk)

            # Непрочитанный фрагмент не должен попасть в vector store.
            assert result.is_useful is False
            assert result.category == (sample_chunk.category or DocCategory.DOCUMENT)
            assert result.confidence == 0.0
            assert "API connection error" in result.reason

    @pytest.mark.asyncio
    async def test_classify_unparsable_reply_drops_chunk(self, classifier, sample_chunk):
        """Прокис от модели вместо JSON отбрасывает фрагмент."""
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _completion("Конечно! Это обычный текст.")

            result = await classifier.classify(sample_chunk)

            assert result.is_useful is False
            assert result.reason

    @pytest.mark.asyncio
    async def test_classify_skips_truncated_chunk(self, classifier_config):
        """Фрагмент короче порога не вызывает модель вовсе."""
        classifier = ChunkClassifier(**{**classifier_config, "trash_size": 200})
        short = RawElement(source="page-3.pdf", content="Стр. 3")

        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            result = await classifier.classify(short)

            mock_create.assert_not_called()
            assert result.is_useful is False
            assert result.reason == "below_trash_size"

    @pytest.mark.asyncio
    async def test_classify_records_token_usage(self, classifier, sample_chunk):
        """Токены классификатора попадают в счётчик cost."""
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _completion(
                '{"is_useful": true, "category": "faq", "confidence": 0.9}'
            )

            assert classifier.usage["total_tokens"] == 0
            await classifier.classify(sample_chunk)
            await classifier.classify(sample_chunk)

            assert classifier.usage["calls"] == 2
            assert classifier.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_classify_with_null_category(self, classifier):
        """Тест классификации с null категорией"""
        chunk_without_category = RawElement(
            source="test.pdf",
            category=None,  # category is None
            content=SAMPLE_CONTENT,
            metadata={}
        )

        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API error")

            result = await classifier.classify(chunk_without_category)

            # Должен быть DOCUMENT как fallback, и фрагмент отброшен
            assert result.category == DocCategory.DOCUMENT
            assert result.is_useful is False

    def test_parse_response_valid_json(self, classifier):
        """Тест парсинга валидного JSON ответа"""
        raw_response = '{"is_useful": true, "category": "paragraph", "confidence": 0.88}'
        result = classifier._parse_response(raw_response)

        assert isinstance(result, Classified)
        assert result.is_useful is True
        assert result.category == DocCategory.DOCUMENT
        assert result.confidence == 0.88

    def test_parse_response_json_with_extra_text(self, classifier):
        """Тест парсинга JSON с дополнительным текстом"""
        raw_response = 'Here is the classification: {"is_useful": true, "category": "section", "confidence": 0.75}'
        result = classifier._parse_response(raw_response)

        assert isinstance(result, Classified)
        assert result.is_useful is True
        assert result.category == DocCategory.DOCUMENT
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
        raw_response = '{"is_useful": true, "category": "glossary", "confidence": 0.9}'
        result = classifier._parse_response(raw_response)

        assert result.is_useful is True
        assert result.category == DocCategory.GLOSSARY
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_classify_logging_on_success(self, classifier, sample_chunk, caplog):
        """Тест логирования при успешной классификации"""
        mock_response = _completion(
            '{"is_useful": true, "category": "section", "confidence": 0.95}'
        )

        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            with caplog.at_level("DEBUG"):
                await classifier.classify(sample_chunk)

                assert "classification_processing_start" in caplog.text

    @pytest.mark.asyncio
    async def test_classify_logging_on_error(self, classifier, sample_chunk, caplog):
        """Тест логирования при ошибке классификации"""
        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Connection timeout")

            with caplog.at_level("ERROR"):
                await classifier.classify(sample_chunk)

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
        mock_response.usage = None

        with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await classifier.classify(sample_chunk)

            # Пустой ответ — это не классификация, фрагмент отбрасывается
            assert result.is_useful is False
            assert result.category == (sample_chunk.category or DocCategory.DOCUMENT)
            assert result.confidence == 0.0
            # Нет usage — счётчик не должен поехать
            assert classifier.usage["calls"] == 0

    def test_excluded_categories_immutable(self, classifier):
        """Тест что excluded_categories не изменяется случайно"""
        # Пытаемся изменить
        classifier.excluded_categories.append(DocCategory.DOCUMENT)

        # Лучше проверить, что KNOWLEDGE_CANVAS всегда исключен
        assert DocCategory.KNOWLEDGE_CANVAS in classifier.excluded_categories


    def test_build_messages_is_system_plus_user(self, classifier, sample_chunk):
        """Провайдер отвергает диалог только из system-сообщений."""
        messages = classifier._build_messages(sample_chunk)

        assert [m["role"] for m in messages] == ["system", "user"]
        system, user = messages
        # Фрагмент — это запрос, он уходит в user и не должен засорять роль.
        assert sample_chunk.content in user["content"]
        assert sample_chunk.content not in system["content"]
        # Роль модели и постановка задачи остаются в system.
        lead = classifier.system_prompt.partition("{chunk_text}")[0].strip()
        assert lead and lead in system["content"]
        assert system["content"].strip()

    def test_build_messages_keeps_category_descriptions(self, classifier, sample_chunk):
        messages = classifier._build_messages(sample_chunk)
        combined = " ".join(m["content"] for m in messages)
        assert classifier.categories_str in combined

    def test_build_messages_handles_override_without_placeholder(
        self, classifier_config, sample_chunk
    ):
        """Шаблон без {chunk_text} не должен терять фрагмент."""
        classifier = ChunkClassifier(
            **{
                **classifier_config,
                "system_prompt": "Just classify. Categories: {category_descriptions}",
            }
        )
        messages = classifier._build_messages(sample_chunk)
        assert [m["role"] for m in messages] == ["system", "user"]
        # Инструкция — в system, а фрагмент обязан уехать в user целиком.
        assert classifier.categories_str in messages[0]["content"]
        assert sample_chunk.content in messages[1]["content"]
        assert sample_chunk.content not in messages[0]["content"]

    def test_build_messages_real_prompt_keeps_json_spec(self, sample_chunk):
        """Штатный промпт не должен терять спецификацию ответа."""
        classifier = ChunkClassifier(
            api_url="http://test-api.com",
            api_key="test-key",
            model="test-model",
            system_prompt=CHUNK_CLASSIFIER_PROMPT,
            category_descriptions={"document": "A formal document"},
            trash_size=0,
        )
        messages = classifier._build_messages(sample_chunk)
        assert [m["role"] for m in messages] == ["system", "user"]
        combined = " ".join(m["content"] for m in messages)
        # Без этих ключей модель отвечает прозой и чанк уходит в `except`.
        assert "is_useful" in combined
        assert sample_chunk.content in messages[1]["content"]
        assert sample_chunk.content not in messages[0]["content"]


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
            },
            trash_size=0,
        )

    @pytest.mark.asyncio
    async def test_classify_different_categories(self, classifier):
        """Тест классификации разных категорий"""
        test_cases = [
            (SAMPLE_CONTENT, "introduction", 0.9),
            (SAMPLE_CONTENT + " Дополнение.", "body", 0.85),
            (SAMPLE_CONTENT + " Итог.", "conclusion", 0.8)
        ]

        for content, expected_category, expected_confidence in test_cases:
            chunk = RawElement(
                source="test.pdf",
                category=DocCategory.DOCUMENT,
                content=content,
                metadata={}
            )

            mock_response = _completion(
                f'{{"is_useful": true, "category": "{expected_category}", '
                f'"confidence": {expected_confidence}}}'
            )

            with patch.object(classifier.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
                mock_create.return_value = mock_response

                result = await classifier.classify(chunk)

                assert result.is_useful is True
                assert result.category == DocCategory(expected_category)
                assert result.confidence == expected_confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
