"""
URL fetching utilities with retry logic and async support
"""
import aiohttp
import asyncio
from typing import Optional, Any
from pathlib import Path
from urllib.parse import urlparse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from loguru import logger

from shared.models.base import IDataFetcher


class URLFetchError(Exception):
    """Custom exception for URL fetching errors"""
    pass


class AsyncURLFetcher(IDataFetcher[bytes]):
    """Async URL fetcher with retry logic and rate limiting"""

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit: Optional[int] = None,
        headers: Optional[dict[str, str]] = None
    ):
        """
        Initialize URL fetcher

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit: Maximum requests per second (optional)
            headers: Default headers for requests
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.headers = headers or {
            "User-Agent": "QuantAI/0.1.0 (Quant Research Platform)"
        }
        self._semaphore: Optional[asyncio.Semaphore] = None
        if rate_limit:
            self._semaphore = asyncio.Semaphore(rate_limit)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def fetch(self, source: str, **kwargs: Any) -> bytes:
        """
        Fetch data from URL with automatic retry

        Args:
            source: URL to fetch
            **kwargs: Additional arguments (headers, params, etc.)

        Returns:
            Response content as bytes

        Raises:
            URLFetchError: If fetch fails after retries
        """
        headers = {**self.headers, **kwargs.get("headers", {})}

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                if self._semaphore:
                    async with self._semaphore:
                        return await self._fetch_with_session(session, source, headers, kwargs)
                else:
                    return await self._fetch_with_session(session, source, headers, kwargs)

            except aiohttp.ClientError as e:
                logger.error(f"Failed to fetch {source}: {e}")
                raise URLFetchError(f"Failed to fetch {source}: {e}") from e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout fetching {source}")
                raise URLFetchError(f"Timeout fetching {source}") from e

    async def _fetch_with_session(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
        kwargs: dict[str, Any]
    ) -> bytes:
        """Internal method to fetch with a session"""
        async with session.get(
            url,
            headers=headers,
            params=kwargs.get("params"),
            allow_redirects=kwargs.get("allow_redirects", True)
        ) as response:
            response.raise_for_status()
            logger.info(f"Successfully fetched {url} (status: {response.status})")
            return await response.read()

    async def fetch_batch(self, sources: list[str], **kwargs: Any) -> list[bytes]:
        """
        Fetch multiple URLs concurrently

        Args:
            sources: List of URLs to fetch
            **kwargs: Additional arguments

        Returns:
            List of response contents
        """
        tasks = [self.fetch(source, **kwargs) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors but return successful results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {sources[i]}: {result}")

        return [r for r in results if not isinstance(r, Exception)]

    async def validate_source(self, source: str) -> bool:
        """
        Validate if URL is accessible with HEAD request

        Args:
            source: URL to validate

        Returns:
            True if accessible, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.head(source, headers=self.headers) as response:
                    return response.status < 400
        except Exception as e:
            logger.warning(f"URL validation failed for {source}: {e}")
            return False

    async def download_file(
        self,
        url: str,
        destination: Path,
        chunk_size: int = 8192
    ) -> Path:
        """
        Download file from URL to local path with streaming

        Args:
            url: URL to download from
            destination: Local file path to save to
            chunk_size: Size of chunks for streaming download

        Returns:
            Path to downloaded file

        Raises:
            URLFetchError: If download fails
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0

                    with open(destination, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.debug(f"Download progress: {progress:.1f}%")

            logger.info(f"Downloaded {url} to {destination} ({downloaded} bytes)")
            return destination

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if destination.exists():
                destination.unlink()
            raise URLFetchError(f"Failed to download {url}: {e}") from e

    async def fetch_json(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """
        Fetch and parse JSON from URL

        Args:
            url: URL to fetch
            **kwargs: Additional arguments

        Returns:
            Parsed JSON as dictionary
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url, headers=self.headers, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

    async def fetch_text(self, url: str, encoding: str = "utf-8", **kwargs: Any) -> str:
        """
        Fetch text content from URL

        Args:
            url: URL to fetch
            encoding: Text encoding
            **kwargs: Additional arguments

        Returns:
            Text content
        """
        content = await self.fetch(url, **kwargs)
        return content.decode(encoding)

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format

        Args:
            url: URL to validate

        Returns:
            True if valid URL format
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class DocumentURLFetcher:
    """Specialized fetcher for documents (PDF, EPUB, etc.)"""

    def __init__(self, cache_dir: Path = Path("./data/cache/documents")):
        """
        Initialize document fetcher

        Args:
            cache_dir: Directory to cache downloaded documents
        """
        self.fetcher = AsyncURLFetcher()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_document(
        self,
        url: str,
        force_download: bool = False
    ) -> Path:
        """
        Fetch document from URL, using cache if available

        Args:
            url: URL of document
            force_download: Force re-download even if cached

        Returns:
            Path to local document file
        """
        # Generate cache filename from URL
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "document"

        # Ensure valid filename
        if not any(filename.endswith(ext) for ext in ['.pdf', '.epub', '.docx', '.txt']):
            filename += '.pdf'  # Default to PDF

        cache_path = self.cache_dir / filename

        # Check cache
        if cache_path.exists() and not force_download:
            logger.info(f"Using cached document: {cache_path}")
            return cache_path

        # Download
        logger.info(f"Downloading document from {url}")
        return await self.fetcher.download_file(url, cache_path)

    async def fetch_multiple_documents(
        self,
        urls: list[str],
        max_concurrent: int = 3
    ) -> list[Path]:
        """
        Fetch multiple documents with concurrency control

        Args:
            urls: List of document URLs
            max_concurrent: Maximum concurrent downloads

        Returns:
            List of paths to downloaded documents
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url: str) -> Path:
            async with semaphore:
                return await self.fetch_document(url)

        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if isinstance(r, Path)]
