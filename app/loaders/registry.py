# app/loaders/registry.py

from loaders.html_loader import HTMLDocumentLoader
from loaders.pdf_loader import PDFDocumentLoader
from loaders.table_loader import TableDocumentLoader
from loaders.word_loader import WordDocumentLoader


from settings import Settings

settings: Settings = Settings()
LOADER_EXTENSION_MAP = {
    "docx": (WordDocumentLoader(settings)),
    "pdf": (PDFDocumentLoader(settings)),
    "csv": (TableDocumentLoader(settings)),
    "xlsx": (TableDocumentLoader(settings)),
    "xls": (TableDocumentLoader(settings)),
    "html": (HTMLDocumentLoader(settings)),
    # "text" обрабатывается особо
}