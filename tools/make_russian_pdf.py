"""Generate a Russian test PDF.

The PDF path in this pipeline is visual: pages are rendered with poppler and
read by the VLM, because a PDF rarely has a usable text layer. So the fixture
has to contain real, visible Cyrillic glyphs rather than a text layer we could
extract with pdfplumber.

Rendering with Pillow and saving through Pillow's PDF writer keeps this to the
dependencies the project already has, and needs no new font tooling: a TTF from
the host's font directory is embedded as-is.

    python tools/make_russian_pdf.py [--out ./data/sample_ru.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAGE_W, PAGE_H = 1240, 1754  # A4 at 150 dpi
MARGIN = 90

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

PAGES: list[list[tuple[str, int]]] = [
    [
        ("Договор поставки № 41-А", 46),
        ("г. Москва, 17 марта 2026 г.", 26),
        ("", 20),
        ("Поставщик: ООО «Северный Путь», ИНН 7701234567, ОГРН 1157746001234", 26),
        ("Адрес: 125167, г. Москва, Ленинградский пр-т, д. 39, стр. 1", 26),
        ("Покупатель: ООО «Восток Трейд», ИНН 7802234567, ОГРН 1127847009876", 26),
        ("Адрес: 191036, г. Санкт-Петербург, ул. Комсомольская, д. 15", 26),
        ("", 20),
        ("1. Предмет договора", 32),
        ("1.1. Поставщик обязуется передать в собственность Покупателя товар,", 26),
        ("указанный в Спецификации (Приложение № 1), а Покупатель обязуется", 26),
        ("оплатить этот товар на условиях, предусмотренных настоящим договором.", 26),
        ("", 20),
        ("1.2. Наименование, количество, цена и срок поставки каждой позиции", 26),
        ("определяются Спецификацией, являющейся неотъемлемой частью договора.", 26),
        ("", 20),
        ("2. Цена и порядок расчётов", 32),
        ("2.1. Цена товара определяется в рублях Российской Федерации и включает", 26),
        ("все применимые налоги, в том числе НДС 20 процентов.", 26),
        ("2.2. Оплата производится безналичным переводом в течение 10 банковских", 26),
        ("дней с момента подписания Спецификации.", 26),
    ],
    [
        ("3. Сроки и порядок поставки", 32),
        ("3.1. Поставка осуществляется складом Поставщика по адресу, указанному", 26),
        ("в Спецификации, в течение 15 рабочих дней с даты оплаты.", 26),
        ("3.2. Право собственности и риск случайной гибели переходят к", 26),
        ("Покупателю в момент подписания товарной накладной.", 26),
        ("", 20),
        ("4. Качество и гарантия", 32),
        ("4.1. Качество товара должно соответствовать ГОСТ 1234-2019.", 26),
        ("4.2. Гарантийный срок составляет 24 месяца с даты поставки.", 26),
        ("4.3. Поставщик обязуется заменить товар ненадлежащего качества", 26),
        ("в течение 15 рабочих дней с момента получения претензии.", 26),
        ("", 20),
        ("5. Ответственность сторон", 32),
        ("5.1. За неисполнение обязательств стороны уплачивают неустойку в размере", 26),
        ("0,1 процента от стоимости непоставленного товара за каждый день просрочки,", 26),
        ("но не более 10 процентов от общей стоимости договора.", 26),
        ("", 20),
        ("6. Прочие условия", 32),
        ("6.1. Договор вступает в силу с момента подписания и действует до", 26),
        ("полного исполнения обязательств сторонами.", 26),
    ],
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise SystemExit(
        "No TrueType font found for Cyrillic rendering. Tried: "
        + ", ".join(FONT_CANDIDATES)
    )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        probe = f"{current} {word}".strip()
        if draw.textlength(probe, font=font) <= width:
            current = probe
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_page(index: int) -> Image.Image:
    image = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(image)
    y = MARGIN

    for text, size in PAGES[index]:
        if not text:
            y += size
            continue
        font = _load_font(size)
        for line in _wrap(draw, text, font, PAGE_W - 2 * MARGIN):
            draw.text((MARGIN, y), line, font=font, fill="black")
            y += int(size * 1.45)

    footer = _load_font(20)
    draw.text(
        (PAGE_W // 2 - 60, PAGE_H - MARGIN),
        f"Страница {index + 1} из {len(PAGES)}",
        font=footer,
        fill=(90, 90, 90),
    )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("./data/sample_ru.pdf"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages = [_render_page(i) for i in range(len(PAGES))]
    pages[0].save(
        args.out,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=pages[1:],
    )
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
