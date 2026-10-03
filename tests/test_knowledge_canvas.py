import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from src.knowledge_canvas_enricher import KnowledgeCanvasEnricher


class TestKnowledgeCanvasEnricher:
    """Тесты для KnowledgeCanvasEnricher"""
    
    @pytest.fixture
    def enricher_config(self):
        """Базовая конфигурация для энричера"""
        return {
            "api_url": "http://test-api.com",
            "api_key": "test-key",
            "model": "gpt-4",
            "system_prompt": "You are a knowledge base expert. Create structured knowledge from the given data."
        }
    
    @pytest.fixture
    def enricher(self, enricher_config):
        """Создание экземпляра энричера"""
        return KnowledgeCanvasEnricher(**enricher_config)
    
    @pytest.fixture
    def sample_data(self):
        """Пример данных для создания базы знаний"""
        return """
        Компания: ООО "ТехноИнновации"
        Основана: 2015 год
        Сфера деятельности: Разработка программного обеспечения
        Ключевые продукты: 
        - CRM система "БизнесКонтроль"
        - Платформа аналитики "DataVision"
        Количество сотрудников: 150
        Головной офис: Москва
        """
    
    @pytest.fixture
    def sample_schema(self):
        """Пример схемы для базы знаний"""
        return {
            "company": {
                "name": "string",
                "founded": "integer",
                "industry": "string",
                "products": "list",
                "employees": "integer",
                "location": "string"
            }
        }
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_success(self, enricher, sample_data):
        """Тест успешного создания базы знаний"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"company": {"name": "ООО ТехноИнновации", "founded": 2015}}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data)
            
            # Проверяем вызов API
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            assert call_args["model"] == enricher.model
            
            # Проверяем сообщения
            messages = call_args["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == enricher.system_prompt
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == sample_data
            
            # Проверяем результат
            assert isinstance(result, str)
            assert "ООО ТехноИнновации" in result
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_with_schema(self, enricher, sample_data, sample_schema):
        """Тест создания базы знаний со схемой"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"company": {"name": "ООО ТехноИнновации", "founded": 2015, "industry": "Software", "products": ["CRM", "Analytics"], "employees": 150, "location": "Moscow"}}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data, sample_schema)
            
            # Проверяем, что схема передается в запросе
            call_args = mock_create.call_args[1]
            user_message = call_args["messages"][1]["content"]
            assert sample_data in user_message
            
            # Результат должен соответствовать схеме
            result_dict = json.loads(result)
            assert "company" in result_dict
            assert "name" in result_dict["company"]
            assert "founded" in result_dict["company"]
            assert "industry" in result_dict["company"]
            assert "products" in result_dict["company"]
            assert "employees" in result_dict["company"]
            assert "location" in result_dict["company"]
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_with_empty_data(self, enricher):
        """Тест с пустыми данными"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"message": "No data provided"}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas("")
            
            call_args = mock_create.call_args[1]
            assert call_args["messages"][1]["content"] == ""
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_api_error(self, enricher, sample_data):
        """Тест обработки ошибки API"""
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API connection error")
            
            # Проверяем, что исключение пробрасывается дальше
            with pytest.raises(Exception) as exc_info:
                await enricher.create_knowledge_canvas(sample_data)
            
            assert "API connection error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_timeout(self, enricher, sample_data):
        """Тест таймаута API"""
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = TimeoutError("Request timeout")
            
            with pytest.raises(TimeoutError) as exc_info:
                await enricher.create_knowledge_canvas(sample_data)
            
            assert "Request timeout" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_response_without_choices(self, enricher, sample_data):
        """Тест обработки ответа без choices"""
        mock_response = Mock()
        mock_response.choices = None
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data)
            
            # Должен вернуть пустую строку
            assert result == ""
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_empty_response(self, enricher, sample_data):
        """Тест с пустым ответом от LLM"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=""))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data)
            
            assert result == ""
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_json_response(self, enricher, sample_data):
        """Тест с JSON ответом от LLM"""
        expected_json = {
            "company": {
                "name": "ООО ТехноИнновации",
                "founded": 2015,
                "industry": "Software Development",
                "products": ["CRM", "Analytics", "Cloud"],
                "employees": 150,
                "location": "Moscow"
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(expected_json)))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data)
            
            # Проверяем, что результат - валидный JSON
            result_dict = json.loads(result)
            assert result_dict == expected_json
            assert result_dict["company"]["name"] == "ООО ТехноИнновации"
            assert result_dict["company"]["founded"] == 2015
            assert len(result_dict["company"]["products"]) == 3
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_with_complex_schema(self, enricher, sample_data):
        """Тест со сложной схемой"""
        complex_schema = {
            "company": {
                "name": "string",
                "founded": "integer",
                "industry": "string",
                "products": ["string"],
                "employees": "integer",
                "location": "string",
                "departments": {
                    "engineering": "integer",
                    "sales": "integer",
                    "marketing": "integer",
                    "hr": "integer"
                },
                "revenue": {
                    "2023": "float",
                    "2024": "float"
                }
            }
        }
        
        expected_response = {
            "company": {
                "name": "ООО ТехноИнновации",
                "founded": 2015,
                "industry": "Software Development",
                "products": ["CRM", "Analytics", "Cloud"],
                "employees": 150,
                "location": "Moscow",
                "departments": {
                    "engineering": 80,
                    "sales": 30,
                    "marketing": 25,
                    "hr": 15
                },
                "revenue": {
                    "2023": 500.5,
                    "2024": 750.3
                }
            }
        }
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=json.dumps(expected_response)))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data, complex_schema)
            
            result_dict = json.loads(result)
            assert "departments" in result_dict["company"]
            assert "engineering" in result_dict["company"]["departments"]
            assert "revenue" in result_dict["company"]
            assert "2023" in result_dict["company"]["revenue"]
            assert "2024" in result_dict["company"]["revenue"]
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_logging_success(self, enricher, sample_data, caplog):
        """Тест логирования при успешном создании"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"result": "success"}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            with caplog.at_level("DEBUG"):
                result = await enricher.create_knowledge_canvas(sample_data)
                
                assert "knowledge_canvas_processing_start" in caplog.text
                assert "knowledge_canvas_processing_done" in caplog.text
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_logging_error(self, enricher, sample_data, caplog):
        """Тест логирования при ошибке"""
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API error")
            
            with pytest.raises(Exception):
                await enricher.create_knowledge_canvas(sample_data)
            
            # Проверяем, что старт был залогирован
            assert "knowledge_canvas_processing_start" in caplog.text
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_large_data(self, enricher):
        """Тест с большим объемом данных"""
        large_data = "Company data: " + "A" * 10000
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"processed": true, "size": 10000}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(large_data)
            
            # Проверяем, что данные переданы полностью
            call_args = mock_create.call_args[1]
            assert call_args["messages"][1]["content"] == large_data
            assert "processed" in result
    
    @pytest.mark.asyncio
    async def test_create_knowledge_canvas_unicode_data(self, enricher):
        """Тест с Unicode данными"""
        unicode_data = """
        Компания: ООО "ТехноИнновации"  
        Продукты: 
        - CRM система "БизнесКонтроль" 
        - Платформа аналитики "DataVision" 
        Адрес: г. Москва, ул. Тверская, д. 25
        """
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"status": "success", "message": "Данные обработаны"}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(unicode_data)
            
            assert "Данные обработаны" in result


class TestKnowledgeCanvasEnricherIntegration:
    """Интеграционные тесты"""
    
    @pytest.fixture
    def enricher(self):
        return KnowledgeCanvasEnricher(
            api_url="http://localhost:8000",
            api_key="test-key",
            model="llama2",
            system_prompt="You are a knowledge extraction expert."
        )
    
    @pytest.mark.asyncio
    async def test_multiple_calls(self, enricher):
        """Тест множественных вызовов"""
        datasets = [
            "Data set 1: Company Alpha",
            "Data set 2: Company Beta",
            "Data set 3: Company Gamma"
        ]
        
        expected_responses = [
            '{"company": "Alpha", "status": "processed"}',
            '{"company": "Beta", "status": "processed"}',
            '{"company": "Gamma", "status": "processed"}'
        ]
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Создаем моки для каждого вызова
            mock_create.side_effect = [
                Mock(choices=[Mock(message=Mock(content=resp))], usage=Mock())
                for resp in expected_responses
            ]
            
            results = []
            for data in datasets:
                result = await enricher.create_knowledge_canvas(data)
                results.append(result)
            
            # Проверяем, что все вызовы были сделаны
            assert mock_create.call_count == 3
            
            # Проверяем результаты
            for i, result in enumerate(results):
                assert json.loads(result)["company"] == datasets[i].split(": ")[1]
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, enricher):
        """Тест восстановления после ошибки"""
        data = "Test data"
        
        # Первый вызов падает, второй успешный
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [
                Exception("Temporary error"),
                Mock(choices=[Mock(message=Mock(content='{"status": "success"}'))], usage=Mock())
            ]
            
            # Первый вызов должен упасть
            with pytest.raises(Exception):
                await enricher.create_knowledge_canvas(data)
            
            # Второй вызов должен быть успешным
            result = await enricher.create_knowledge_canvas(data)
            assert json.loads(result)["status"] == "success"
            
            assert mock_create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self, enricher):
        """Тест конкурентных вызовов"""
        import asyncio
        
        data = "Test data"
        mock_response = Mock(choices=[Mock(message=Mock(content='{"status": "success"}'))], usage=Mock())
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Запускаем 5 конкурентных вызовов
            tasks = [enricher.create_knowledge_canvas(data) for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
            # Проверяем, что все вызовы были сделаны
            assert mock_create.call_count == 5
            assert len(results) == 5
            for result in results:
                assert json.loads(result)["status"] == "success"


class TestKnowledgeCanvasEnricherEdgeCases:
    """Тесты для пограничных случаев"""
    
    @pytest.fixture
    def enricher(self):
        return KnowledgeCanvasEnricher(
            api_url="http://test-api.com",
            api_key="test-key",
            model="gpt-4",
            system_prompt="System prompt"
        )
    
    @pytest.mark.asyncio
    async def test_none_data(self, enricher):
        """Тест с None в качестве данных"""
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content='{"error": "No data provided"}'))
            ]
            mock_response.usage = Mock()
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(None)
            
            # Проверяем, что None преобразован в строку
            call_args = mock_create.call_args[1]
            assert call_args["messages"][1]["content"] == "None"
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_special_characters_in_data(self, enricher):
        """Тест со специальными символами"""
        data = "Special chars: \n\t\r\b\f\\\"\'"
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"processed": true, "chars": "escaped"}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(data)
            
            call_args = mock_create.call_args[1]
            assert call_args["messages"][1]["content"] == data
            assert "processed" in result
    
    @pytest.mark.asyncio
    async def test_empty_schema(self, enricher, sample_data):
        """Тест с пустой схемой"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"result": "processed"}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await enricher.create_knowledge_canvas(sample_data, {})
            
            assert "result" in result
    
    @pytest.mark.asyncio
    async def test_invalid_json_in_schema(self, enricher, sample_data):
        """Тест с невалидной JSON схемой"""
        invalid_schema = "not a dict"
        
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"error": "Invalid schema"}'))
        ]
        mock_response.usage = Mock()
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Схема невалидна, но мы все равно передаем ее (как есть)
            result = await enricher.create_knowledge_canvas(sample_data, invalid_schema)
            
            call_args = mock_create.call_args[1]
            # Схема должна быть преобразована в строку
            assert str(invalid_schema) in call_args["messages"][1]["content"]
    
    @pytest.mark.asyncio
    async def test_max_token_limit(self, enricher):
        """Тест с данными, превышающими лимит токенов"""
        huge_data = "A" * 100000  # Очень большой текст
        
        with patch.object(enricher.openai_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            # Имитируем ошибку превышения лимита
            mock_create.side_effect = Exception("This model's maximum context length is 4096 tokens")
            
            with pytest.raises(Exception) as exc_info:
                await enricher.create_knowledge_canvas(huge_data)
            
            assert "maximum context length" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])