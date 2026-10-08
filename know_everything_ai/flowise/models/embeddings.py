from typing import Any

from .base import BaseConfig


class EmbeddingConfig(BaseConfig):
    """
    Конфигурация для embedding-модели.

    Поддерживаемые провайдеры:
        - OpenAI (openAIEmbeddings)
        - Azure OpenAI (azureOpenAIEmbeddings)
        - LocalAI (localAIEmbeddings)
        - HuggingFace Inference (huggingFaceInferenceEmbeddings)
        - и другие (можно указать name произвольно)
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)


    # ---------- LocalAI ----------
    @classmethod
    def localai(
        cls,
        base_url: str,
        model: str,
        **kwargs
    ) -> "EmbeddingConfig":
        config = {"baseURL": base_url, "model": model, **kwargs}
        return cls("localAIEmbeddings", config)

    # ---------- HuggingFace Inference  ----------
    @classmethod
    def huggingface_inference(
        cls,
        model: str,
        credential: str | None = None,
        api_url: str | None = None,
        **kwargs
    ) -> "EmbeddingConfig":
        """
        Hugging Face Inference API для эмбеддингов.
        Документация: https://huggingface.co/docs/api-inference/index

        :param model: Название модели, например 'sentence-transformers/all-MiniLM-L6-v2'
        :param credential: ID ключа Hugging Face в Flowise
        :param api_url: Пользовательский URL
        """
        config = {"modelName": model, **kwargs}
        if credential:
            config["credential"] = credential
        if api_url:
            config["endpoint"] = api_url
        return cls("huggingFaceInferenceEmbeddings", config)

    # ---------- Кастомный провайдер (просто по имени) ----------
    @classmethod
    def custom(cls, name: str, config: dict[str, Any] | None = None) -> "EmbeddingConfig":
        """Для любого другого эмбеддера, не описанного выше."""
        return cls(name, config or {})
