"""
Очистка текста от технического мусора, рекламных блоков, 
нормализация пробелов и пустых строк.
"""
import re
from typing import List, Optional, Pattern

# Типичные мусорные строки (можно расширять под конкретных клиентов)
DEFAULT_NOISE_PATTERNS: List[Pattern] = [
    re.compile(r"^\s*(Поделиться|Share|Tweet|Pin|Email)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(Читайте также|Related articles|Вам также будет интересно)\s*:?", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}.*$"),                     # копирайты
    re.compile(r"^\s*All rights reserved\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Мы используем cookie.*$", re.IGNORECASE),
    re.compile(r"^\s*Наш сайт использует файлы cookie.*$", re.IGNORECASE),
    re.compile(r"^\s*Политика конфиденциальности\s*$", re.IGNORECASE),
    re.compile(r"^\s*Privacy Policy\s*$", re.IGNORECASE),
    re.compile(r"^\s*(←|&larr;|Назад|Back to top)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Подписаться на рассылку.*$", re.IGNORECASE),
    re.compile(r"^\s*Subscribe to our newsletter.*$", re.IGNORECASE),
]

def clean_text(
    text: str,
    custom_noise_patterns: Optional[List[Pattern]] = None,
    remove_extra_empty_lines: bool = True,
    normalize_unicode: bool = True,
    collapse_spaces: bool = True,
    min_line_length: int = 3,
) -> str:
    """
    Базовая очистка текста перед дальнейшей обработкой.

    :param text: исходный текст (обычно после конвертации в Markdown)
    :param custom_noise_patterns: список дополнительных регулярок для удаления целых строк
    :param remove_extra_empty_lines: схлопнуть несколько пустых строк в одну
    :param normalize_unicode: привести Unicode-символы к нормальной форме (NFC)
    :param collapse_spaces: заменить множественные пробелы/табы на одиночный пробел (но не трогать переводы строк)
    :param min_line_length: удалить строки короче этого значения (если они не часть таблицы)
    :return: очищенный текст
    """
    if not text:
        return ""

    # 1. Нормализация Unicode (каноническая композиция)
    if normalize_unicode:
        import unicodedata
        text = unicodedata.normalize('NFC', text)

    # 2. Удаление BOM и других невидимых управляющих символов (кроме стандартных пробельных)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = text.replace('\ufeff', '')
    lines = text.splitlines()
    cleaned_lines = []

    # Соберём полный список шумовых паттернов
    all_patterns = DEFAULT_NOISE_PATTERNS[:]
    if custom_noise_patterns:
        all_patterns.extend(custom_noise_patterns)

    for line in lines:
        stripped = line.strip()

        # 3. Пропускаем полностью пустые строки? Пока сохраним (будем схлопывать позже)
        # Но отметим их отдельно, чтобы потом обработать
        if not stripped:
            cleaned_lines.append("")
            continue

        # 4. Проверка на шумовые паттерны (удаляем всю строку)
        if any(pattern.search(stripped) for pattern in all_patterns):
            continue  # заменяем на пустую строку (потом схлопнем)

        # 5. Удаление слишком коротких строк, если это не заголовок и не элемент таблицы/списка
        if len(stripped) < min_line_length:
            # Не трогаем заголовки (#, ##), строки таблиц (|) и элементы списков (-, *, 1.)
            if not re.match(r'^\s*(#|-|\*|\d+\.|\|)', line):
                continue

        # 6. Схлопывание множественных пробелов внутри строки (но не трогаем отступы для списков)
        if collapse_spaces:
            # Сохраним ведущие пробелы для структуры
            left_spaces = len(line) - len(line.lstrip(' '))
            # Убираем табы (заменяем на пробел), удаляем двойные пробелы
            content = line.lstrip(' ')
            content = re.sub(r'[ \t]+', ' ', content)
            line = ' ' * left_spaces + content
        else:
            # Просто заменяем табы на пробел
            line = line.replace('\t', ' ')

        cleaned_lines.append(line)

    # 7. Склеиваем обратно и схлопываем множественные пустые строки
    text = '\n'.join(cleaned_lines)
    if remove_extra_empty_lines:
        text = re.sub(r'\n{3,}', '\n\n', text)  # максимум одна пустая строка подряд
        text = text.strip('\n')

    return text