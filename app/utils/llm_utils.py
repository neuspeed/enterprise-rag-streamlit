import abc

import openai

class LLMDriven(abc.ABC):
    """Класс для задач связанных с взаимодействием с LLM. 
    Содержит базовые составляющие, необходимые для подключения модели в работу"""
    def __init__(self, 
                api_url: str,
                api_key: str,
                model: str,
                system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.openai_client: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=3600
        )