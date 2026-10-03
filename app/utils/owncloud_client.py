# owncloud_client.py
import aiohttp
import aiofiles               
import os
from urllib.parse import urlparse
from yarl import URL
import structlog
from pydantic import HttpUrl

log = structlog.get_logger("owncloud")


class OwnCloudClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: int = 120,
    ):
        """
        base_url – корень WebDAV, например
        https://cloud.infercom.one/remote.php/webdav/
        """
        # Гарантируем один завершающий слеш
        self.base_url = base_url.rstrip("/") + "/"
        self.auth = aiohttp.BasicAuth(username, password)
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #
    async def download_to_file(self, remote_url: HttpUrl, dest_path: str) -> None:
        """
        remote_url – полная ссылка, полученная в поле `file`.
        dest_path  – путь в контейнере, куда сохраняем.
        """
        # 1️⃣ Приводим к относительному пути внутри WebDAV
        rel_path = self._make_relative(remote_url)

        # 2️⃣ Формируем окончательный URL (корректно работает как с
        #    абсолютным, так и с относительным `remote_url`)
        download_url = (URL(self.base_url) / rel_path).human_repr()
        log.debug("download_start", url=download_url, dest=dest_path)

        # 3️⃣ Убедимся, что каталог назначения существует
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        # 4️⃣ Скачиваем и записываем асинхронно
        async with aiohttp.ClientSession(
            auth=self.auth,
            timeout=self.timeout,
        ) as sess:
            try:
                async with sess.get(download_url) as resp:
                    resp.raise_for_status()

                    # Асинхронно открываем файл и пишем кусками
                    async with aiofiles.open(dest_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            await f.write(chunk)          # ← важно await
            except aiohttp.ClientError as exc:
                # Ошибки HTTP / сетевые – логируем и пробрасываем дальше
                log.error(
                    "download_failed_http",
                    url=download_url,
                    dest=dest_path,
                    exc_info=exc,
                )
                raise
            except OSError as exc:
                # Ошибки файловой системы (нет прав, диск заполнен и т.п.)
                log.error(
                    "download_failed_io",
                    url=download_url,
                    dest=dest_path,
                    exc_info=exc,
                )
                raise

        # 5️⃣ Всё успешно завершилось → фиксируем в логе
        log.info("download_done", url=download_url, dest=dest_path)

    # ------------------------------------------------------------------ #
    # PRIVATE HELPERS
    # ------------------------------------------------------------------ #
    def _make_relative(self, url: HttpUrl) -> str:
        """
        Превращаем любой URL (полный, с доменом, с query‑string) в
        относительный путь внутри WebDAV.
        """
        # `url` уже является строкой, но делаем явный `str()` – на случай,
        # если кто‑то передаст `bytes`.
        parsed = urlparse(str(url))

        # Если в URL уже есть базовый путь – отрежем его.
        prefix = "/remote.php/webdav/"
        if parsed.path.startswith(prefix):
            # Оставляем всё, что после префикса, и убираем ведущие слеши.
            return parsed.path[len(prefix) :].lstrip("/")

        # Иначе считаем, что путь уже относительный.
        # Убираем возможные ведущие слеши, но оставляем query‑string,
        # если он был (в WebDAV обычно не нужен, но сохраняем для
        # совместимости).
        return parsed.path.lstrip("/")
