"""
Универсальная оценка количества токенов в строке.
Использует tiktoken (если доступен) с кодировкой cl100k_base,
либо эвристическую формулу для смешанного текста.
"""
from __future__ import annotations
import re
from typing import Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _ENCODING = None


def estimate_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Оценивает количество токенов в переданном тексте.

    :param text: входной текст
    :param model_name: опциональное имя модели (игнорируется, если tiktoken недоступен)
    :return: примерное число токенов
    """
    if not text:
        return 0

    if _TIKTOKEN_AVAILABLE and _ENCODING is not None:
        # Если указана конкретная модель, можно попытаться получить её кодировку
        if model_name:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = _ENCODING  # fallback на универсальную
        else:
            enc = _ENCODING
        return len(enc.encode(text))

    # Эвристический метод (fallback)
    # 1 токен ~ 4 символа для английского, ~3.2 для русского; усредним 3.6
    # + добавим поправку на слова (чтобы не недооценить короткие строки)
    # Итоговая оценка: max(длина/3.6, кол-во слов * 1.3)
    char_count = len(text)
    word_count = len(re.findall(r'\w+', text, re.UNICODE))

    # Грубая оценка по символам с учётом кириллицы
    # Если в тексте есть кириллические символы, используем коэффициент 3.2, иначе 3.8
    has_cyrillic = bool(re.search(r'[а-яёА-ЯЁ]', text))
    char_per_token = 3.2 if has_cyrillic else 3.8
    char_estimate = char_count / char_per_token

    # Оценка по словам: в среднем 1.3 токена на слово для английского, 1.8 для русского
    word_multiplier = 1.8 if has_cyrillic else 1.3
    word_estimate = word_count * word_multiplier

    # Берём среднее значение, округляем вверх
    return max(1, int((char_estimate + word_estimate) / 2) + 1)