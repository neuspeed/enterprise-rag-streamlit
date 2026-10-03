# HTML Document Loader

import os
import re
import requests 
import structlog

from loaders.base import BaseLoader
from parsers.html_parser import HTMLParser

log = structlog.get_logger("html_loader")

class HTMLDocumentLoader(BaseLoader):
    def __init__(self, settings):
        self.settings = settings
        self.html_content = None
    
    def validate(self):
        pass
    
    async def load(self, source):
        """
        Load HTML document data from the given source.

        Args:
            source: The source from which to load HTML document data. This could be a file path, URL, or any other identifier.
        """  
        try:  
            if self._is_url(source):
                # Load HTML from URL
                log.debug("processing_html_from_url", source=source)
                self.html_content = self._load_from_url(source)
            else:
                # Load HTML from file
                log.debug("processing_html_from_file", source=source)
                self.html_content = self._load_from_file(source)
        except Exception as e:
            print(e)
        
    
    def _is_url(self, s: str) -> bool:
        """
        Проверка, что строка выглядит как URL.
        Принимает как «чистый» URL, так и экранированный (https:\/\/...).
        """
        # Убираем экранирование \/
        cleaned = s.replace('\\/', '/')
        # Простейшее регулярное выражение – начинается с http(s)://
        pattern = r'^https?://'
        return bool(re.search(pattern, cleaned))

    def _load_from_url(self, url: str) -> str:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://google.com'
        }
        proxies = {
            "https": os.environ.get("PROXY")
        }
        log.debug("request_html", url=url)
        response = requests.get(url, timeout=30, headers=headers, proxies=proxies)
        response.raise_for_status()  # Проверяем успешность запроса
        return response.text

    def _load_from_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
        
    async def transform(self, data_path):
        """
        Transform the loaded HTML document data into the desired format.

        Args:
            data_path: The path to document for transform.

        Returns:
            The transformed data.
        """
        await self.load(data_path)
        return await HTMLParser().parse(self.html_content, source=data_path)