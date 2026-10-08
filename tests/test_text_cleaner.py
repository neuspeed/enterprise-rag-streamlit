import re

import pytest

from know_everything_ai.cleaners.text_cleaner import clean_text


class TestCleanText:
    """Тесты для функции clean_text"""

    def test_empty_text(self):
        """Тест с пустым текстом"""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text("\n\n") == ""
        assert clean_text(None) == ""

    def test_basic_cleaning(self):
        """Тест базовой очистки текста"""
        text = "Hello   world!  This  is  a  test."
        result = clean_text(text, collapse_spaces=True)
        assert result == "Hello world! This is a test."

    def test_unicode_normalization(self):
        """Тест нормализации Unicode"""
        text = "Café résumé"
        result = clean_text(text, normalize_unicode=True)
        assert result == "Café résumé"

        text = "Cafe\u0301"
        result = clean_text(text, normalize_unicode=True)
        import unicodedata
        expected = unicodedata.normalize('NFC', text)
        assert result == expected

    def test_remove_bom_and_control_chars(self):
        """Тест удаления BOM и управляющих символов"""
        text = "\ufeffHello\x01world\x02!"
        result = clean_text(text)
        assert result == "Helloworld!"

    def test_noise_patterns_default(self):
        """Тест удаления дефолтных шумовых паттернов"""
        noise_texts = [
            "Share",
            "Поделиться",
            "Tweet",
            "Pin",
            "Email",
            "Читайте также",
            "Related articles",
            "Вам также будет интересно:",
            "© 2024 All rights reserved.",
            "All rights reserved.",
            "Мы используем cookie",
            "Наш сайт использует файлы cookie",
            "Политика конфиденциальности",
            "Privacy Policy",
            "←",
            "&larr;",
            "Назад",
            "Back to top",
            "Подписаться на рассылку",
            "Subscribe to our newsletter",
        ]

        for noise in noise_texts:
            result = clean_text(noise)
            assert result == "", f"Не удален шум: {noise}"

    def test_noise_patterns_with_context(self):
        """Тест удаления шума в контексте нормального текста"""
        text = """Важный контент.
        Share
        Продолжение важного контента.
        © 2024 Все права защищены.
        Конец статьи."""

        result = clean_text(text)
        expected = "Важный контент.\nПродолжение важного контента.\nКонец статьи."
        assert " ".join(result.split()) == " ".join(expected.split())

    def test_custom_noise_patterns(self):
        """Тест использования кастомных шумовых паттернов"""
        custom_patterns = [
            re.compile(r"^\s*Реклама\s*$", re.IGNORECASE),
            re.compile(r"^\s*Спонсор\s*$", re.IGNORECASE),
        ]

        text = """Обычный текст.
        Реклама
        Еще текст.
        Спонсор
        Финальный текст."""

        result = clean_text(text, custom_noise_patterns=custom_patterns)
        expected = "Обычный текст.\nЕще текст.\nФинальный текст."
        assert " ".join(result.split()) == " ".join(expected.split())

    def test_remove_extra_empty_lines(self):
        """Тест удаления лишних пустых строк"""
        text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
        result = clean_text(text, remove_extra_empty_lines=True)
        assert result == "Line 1\n\nLine 2\n\nLine 3"

        result_no_collapse = clean_text(text, remove_extra_empty_lines=False)
        assert result_no_collapse == text

    def test_collapse_spaces(self):
        """Тест схлопывания пробелов"""
        text = "Hello    world!  This  is   a   test."
        result = clean_text(text, collapse_spaces=True)
        assert result == "Hello world! This is a test."

        result_no_collapse = clean_text(text, collapse_spaces=False)
        assert result_no_collapse == text

    def test_preserve_indentation(self):
        """Тест сохранения отступов для списков"""
        text = "    Item 1\n    Item 2\n        Subitem 2.1\n    Item 3"
        result = clean_text(text, collapse_spaces=True)
        assert result == "    Item 1\n    Item 2\n        Subitem 2.1\n    Item 3"

    def test_min_line_length(self):
        """Тест удаления коротких строк"""
        text = "Short\nA\nB\nLonger text\nC\nNormal length"
        result = clean_text(text, min_line_length=3)
        expected = "Short\nLonger text\nNormal length"
        assert result == expected

    def test_min_line_length_preserve_headers(self):
        """Тест сохранения коротких заголовков"""
        text = "# Title\nA\n## Subtitle\nB\nC"
        result = clean_text(text, min_line_length=3)
        expected = "# Title\n## Subtitle"
        assert result == expected

    def test_min_line_length_preserve_lists(self):
        """Тест сохранения коротких элементов списков"""
        text = "- Item\n* Item\n1. Item\n2. A\n- B"
        result = clean_text(text, min_line_length=3)
        expected = "- Item\n* Item\n1. Item\n2. A\n- B"
        assert result == expected

    def test_min_line_length_preserve_tables(self):
        """Тест сохранения коротких строк таблиц"""
        text = "| A | B |\n| 1 | 2 |\n| 3 | 4 |"
        result = clean_text(text, min_line_length=3)
        assert result == text

    def test_mixed_content(self):
        """Тест смешанного контента"""
        text = """
        # Заголовок

        Это важный текст с  множественными   пробелами.

        Share

        Продолжение текста.

        © 2024 Все права защищены.

        ## Подзаголовок

        - Список 1
        - Список 2

        В конце текст.
        """

        result = clean_text(text)

        assert "Share" not in result
        assert "© 2024" not in result
        assert "Все права защищены" not in result
        assert "# Заголовок" in result
        assert "## Подзаголовок" in result
        assert "- Список 1" in result
        assert "- Список 2" in result
        assert "Это важный текст" in result
        assert "Продолжение текста" in result
        assert "В конце текст" in result
        assert "\n\n\n" not in result

    def test_no_modification_of_valid_text(self):
        """Тест, что валидный текст не изменяется"""
        text = """# Заголовок

Это обычный текст с нормальными пробелами.

- Список 1
- Список 2

Обычный абзац."""

        result = clean_text(text)
        assert result == text

    def test_tabs_conversion(self):
        """Тест преобразования табов"""
        text = "Hello\tworld!\tThis\tis\ta\ttest."
        result = clean_text(text, collapse_spaces=False)
        assert result == "Hello world! This is a test."

        result2 = clean_text(text, collapse_spaces=True)
        assert result2 == "Hello world! This is a test."


    def test_preserve_structure(self):
        """Тест сохранения структуры документа"""
        text = """Chapter 1

Section 1.1

Paragraph text here.

Section 1.2

More text."""

        result = clean_text(text)
        assert "Chapter 1" in result
        assert "Section 1.1" in result
        assert "Paragraph text here" in result
        assert "Section 1.2" in result
        assert "More text" in result

    def test_real_world_example(self):
        """Тест на реальном примере"""
        text = """Важная статья

Поделиться

В этой статье мы рассмотрим важные вопросы.

Tweet

© 2024 Все права защищены.

Читайте также другие статьи.

Политика конфиденциальности"""

        result = clean_text(text)
        expected = "Важная статья\nВ этой статье мы рассмотрим важные вопросы."
        assert " ".join(result.split()) == " ".join(expected.split())

    def test_combined_parameters(self):
        """Тест комбинации всех параметров"""
        text = """
        \ufeffЗаголовок

        Текст   с   множественными   пробелами.

        Share

        Текст с  акцентом\u0301.

        © 2024

        - Список
        - Еще список

        Короткий
        """

        result = clean_text(
            text,
            custom_noise_patterns=[re.compile(r"^Короткий$")],
            remove_extra_empty_lines=True,
            normalize_unicode=True,
            collapse_spaces=True,
            min_line_length=5
        )

        assert "Share" not in result
        assert "© 2024" not in result
        assert "Короткий" not in result
        assert "акцентом" in result
        assert "Заголовок" in result
        assert "Текст с множественными пробелами" in result
        assert "Текст с акцентом" in result
        assert "- Список" in result
        assert "- Еще список" in result
        assert "\n\n\n" not in result

    def test_case_insensitive_patterns(self):
        """Тест регистронезависимых паттернов"""
        text = "share\nSHARE\nShare\nsHaRe"
        result = clean_text(text)
        assert result == ""

    def test_patterns_with_whitespace(self):
        """Тест паттернов с разными пробельными символами"""
        text = "  Share  \n\tShare\t\n  Share\n  Поделиться  "
        result = clean_text(text)
        assert result == ""

    def test_remove_only_whole_lines(self):
        """Тест удаления только целых строк"""
        text = "This is a Share button\nShare\nNot share"
        result = clean_text(text)
        assert "This is a Share button" in result
        assert "Not share" in result
        assert result.strip() == "This is a Share button\nNot share"


class TestCleanTextEdgeCases:
    """Тесты для пограничных случаев"""

    def test_text_with_only_noise(self):
        """Текст, состоящий только из шума"""
        text = "Share\nTweet\n© 2024\nПоделиться"
        result = clean_text(text)
        assert result == ""

    def test_text_with_only_empty_lines(self):
        """Текст только из пустых строк"""
        text = "\n\n\n\n"
        result = clean_text(text, remove_extra_empty_lines=True)
        assert result == ""

    def test_very_long_text(self):
        """Тест на очень длинном тексте"""
        long_text = "a" * 10000 + "   " + "b" * 10000
        result = clean_text(long_text, collapse_spaces=True)
        expected = "a" * 10000 + " " + "b" * 10000
        assert result == expected

    def test_special_characters(self):
        """Тест специальных символов"""
        text = "Hello &amp; world! <html> tags &copy;"
        result = clean_text(text)
        assert result == text

    def test_numbers_only(self):
        """Тест с числами"""
        text = "123\n456\n789"
        result = clean_text(text, min_line_length=3)
        assert result == "123\n456\n789"

        result2 = clean_text(text, min_line_length=4)
        assert result2 == ""

    def test_multiline_noise(self):
        """Тест шума, занимающего несколько строк"""
        text = """Line 1
        Copyright notice
        Line 2
        More copyright
        Line 3"""

        custom_patterns = [
            re.compile(r"Copyright.*notice", re.IGNORECASE | re.DOTALL),
        ]

        result = clean_text(text, custom_noise_patterns=custom_patterns)
        assert "Copyright notice" not in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "More copyright" in result
        assert "Line 3" in result


class TestCleanTextIntegration:
    """Интеграционные тесты"""

    def test_real_article(self):
        """Тест на реальной статье"""
        article = """
        Как использовать Python для анализа данных

        Поделиться

        Python - мощный язык программирования для анализа данных.

        Tweet

        В этой статье мы рассмотрим основные библиотеки.

        © 2024 Все права защищены.

        Читайте также: другие статьи по теме.

        ## Установка Python

        Установите Python с официального сайта.

        Политика конфиденциальности
        """

        result = clean_text(article)

        assert "Как использовать Python для анализа данных" in result
        assert "Python - мощный язык программирования" in result
        assert "В этой статье мы рассмотрим основные библиотеки" in result
        assert "## Установка Python" in result
        assert "Установите Python с официального сайта" in result
        assert "Поделиться" not in result
        assert "Tweet" not in result
        assert "Все права защищены" not in result
        assert "Читайте также" not in result
        assert "Политика конфиденциальности" not in result


if __name__ == "main":
    pytest.main([__file__, "-v", "--tb=short"])
