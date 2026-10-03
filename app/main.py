
import asyncio
from pathlib import Path

from settings import Settings

from parsers.table_parser import TableParser
from parsers.word_parser import WordParser
from loaders.pdf_loader import PDFDocumentLoader
from loaders.html_loader import HTMLDocumentLoader

from cleaners.text_cleaner import clean_text
from utils.token_counter import estimate_tokens

file_path = Path("../test/test.pdf")
website_url = "https://docs.usedesk.ru/article/1678"


# if file_path.exists():
#     pdf_parser = PDFParser()
#     print(pdf_parser.parse(file_path))
# else:
#     print(f"Файл не найден: {file_path.absolute()}")
#     print("Доступные файлы в папке test:")
#     if Path("test").exists():
#         for f in Path("test").iterdir():
#             print(f"  - {f.name}")


async def main():
    loader = HTMLDocumentLoader(Settings())
    result = await loader.transform(website_url)
    print(result)
    token_counter = 0
    for entity in result:
        print(f"Content length")
        cleaned_text = clean_text(entity.content)
        token_counter += estimate_tokens(cleaned_text)
        print(f"symbols uncleaned: {len(entity.content)}") 
        print(f"symbols cleaned: {len(cleaned_text)}")
        print(f"tokens: {token_counter}")
    



asyncio.run(main())