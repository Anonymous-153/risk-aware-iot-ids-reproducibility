from __future__ import annotations

import argparse
import http.client
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from typing import BinaryIO


ROOT_URL = "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/"
DEFAULT_SUFFIXES = (".csv", ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".tgz")
CHUNK_SIZE = 1024 * 1024


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class DatasetCrawler:
    root_url: str = ROOT_URL
    allowed_suffixes: tuple[str, ...] = DEFAULT_SUFFIXES
    include_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = self.root_url if self.root_url.endswith("/") else f"{self.root_url}/"
        object.__setattr__(self, "root_url", normalized)
        object.__setattr__(
            self,
            "allowed_suffixes",
            tuple(suffix.lower() for suffix in self.allowed_suffixes),
        )
        object.__setattr__(
            self,
            "include_prefixes",
            tuple(_normalize_dataset_path(prefix) for prefix in self.include_prefixes if prefix.strip()),
        )

    def discover_targets(self, html: str, page_url: str) -> tuple[list[str], list[str]]:
        extractor = LinkExtractor()
        extractor.feed(html)
        downloads: list[str] = []
        pages: list[str] = []
        seen_downloads: set[str] = set()
        seen_pages: set[str] = set()

        for href in extractor.hrefs:
            target = urllib.parse.urljoin(page_url, href)
            parsed = urllib.parse.urlparse(target)
            normalized = urllib.parse.urlunparse(parsed._replace(fragment=""))
            if not self._is_internal(normalized):
                continue
            path = urllib.parse.unquote(parsed.path).lower()
            if self._is_download_file(parsed) or path.endswith(self.allowed_suffixes):
                if not self._matches_include_prefix(_logical_path(parsed), allow_parent=False):
                    continue
                if normalized not in seen_downloads:
                    downloads.append(normalized)
                    seen_downloads.add(normalized)
            elif self._looks_like_browse_page(parsed):
                if not self._matches_include_prefix(_logical_path(parsed), allow_parent=True):
                    continue
                if normalized not in seen_pages:
                    pages.append(normalized)
                    seen_pages.add(normalized)

        return downloads, pages

    def _is_internal(self, url: str) -> bool:
        return url.startswith(self.root_url)

    @staticmethod
    def _looks_like_browse_page(parsed: urllib.parse.ParseResult) -> bool:
        path = urllib.parse.unquote(parsed.path).lower()
        return path.endswith("/browse.php") or path.endswith("/browse")

    def _is_download_file(self, parsed: urllib.parse.ParseResult) -> bool:
        path = urllib.parse.unquote(parsed.path).lower()
        if not path.endswith("/download.php"):
            return False
        query = urllib.parse.parse_qs(parsed.query)
        files = query.get("file", [])
        return any(urllib.parse.unquote(file).lower().endswith(self.allowed_suffixes) for file in files)

    def _matches_include_prefix(self, path: str, *, allow_parent: bool) -> bool:
        if not self.include_prefixes:
            return True
        normalized = _normalize_dataset_path(path)
        if not normalized:
            return True
        for prefix in self.include_prefixes:
            if normalized == prefix or normalized.startswith(f"{prefix}/"):
                return True
            if allow_parent and prefix.startswith(f"{normalized}/"):
                return True
        return False


def _logical_path(parsed: urllib.parse.ParseResult) -> str:
    path = urllib.parse.unquote(parsed.path)
    name = Path(path).name.lower()
    query = urllib.parse.parse_qs(parsed.query)
    if name == "download.php" and query.get("file"):
        return urllib.parse.unquote(query["file"][0])
    if name == "browse.php":
        return urllib.parse.unquote(query.get("p", [""])[0])
    root = urllib.parse.urlparse(ROOT_URL).path
    if path.startswith(root):
        return path[len(root) :]
    return path


def _normalize_dataset_path(path: str) -> str:
    return "/".join(part for part in urllib.parse.unquote(path).replace("\\", "/").strip("/").split("/") if part)


@dataclass(frozen=True)
class RegistrationInfo:
    first_name: str
    last_name: str
    email: str
    institution: str
    job_title: str
    country: str

    def as_form_data(self) -> bytes:
        return urllib.parse.urlencode(
            {
                "first_name": self.first_name,
                "last_name": self.last_name,
                "email": self.email,
                "institution": self.institution,
                "job_title": self.job_title,
                "country": self.country,
            }
        ).encode("utf-8")


def parse_registration_response(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("CIC registration did not return JSON.") from exc

    if not payload.get("ok"):
        message = payload.get("message") or "CIC registration failed."
        raise RuntimeError(str(message))
    return str(payload.get("redirect") or "browse.php")


def local_path_for_url(url: str, destination: Path, root_url: str = ROOT_URL) -> Path:
    root = urllib.parse.urlparse(root_url if root_url.endswith("/") else f"{root_url}/")
    parsed = urllib.parse.urlparse(url)
    root_path = urllib.parse.unquote(root.path)
    target_path = urllib.parse.unquote(parsed.path)
    if not target_path.startswith(root_path):
        raise ValueError(f"download URL is outside dataset root: {url}")

    rel_path = target_path[len(root_path) :].strip("/")
    if rel_path.lower() == "download.php":
        files = urllib.parse.parse_qs(parsed.query).get("file", [])
        if not files:
            raise ValueError(f"download URL is missing the file parameter: {url}")
        rel_path = urllib.parse.unquote(files[0]).strip("/")
    if not rel_path:
        raise ValueError(f"download URL does not contain a file name: {url}")
    parts = Path(rel_path).parts
    if any(part in {"..", ""} for part in parts):
        raise ValueError(f"download URL contains an unsafe path: {url}")
    return destination.joinpath(*parts)


def build_opener(proxy: str | None, insecure_tls: bool) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPCookieProcessor(CookieJar())]
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    if insecure_tls:
        handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    return urllib.request.build_opener(*handlers)


def request_bytes(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    timeout: int,
    data: bytes | None = None,
) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
    if data is not None:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with opener.open(req, timeout=timeout) as response:
        return response.read()


def register(opener: urllib.request.OpenerDirector, info: RegistrationInfo, timeout: int) -> str:
    request_bytes(opener, ROOT_URL, timeout=timeout)
    raw = request_bytes(opener, urllib.parse.urljoin(ROOT_URL, "insert.php"), timeout=timeout, data=info.as_form_data())
    return parse_registration_response(raw)


def crawl_downloads(
    opener: urllib.request.OpenerDirector,
    start_url: str,
    crawler: DatasetCrawler,
    *,
    max_pages: int,
    timeout: int,
) -> list[str]:
    pending = [start_url]
    seen_pages: set[str] = set()
    seen_downloads: set[str] = set()
    downloads: list[str] = []

    while pending:
        page_url = pending.pop(0)
        if page_url in seen_pages:
            continue
        if len(seen_pages) >= max_pages:
            raise RuntimeError(f"stopped after {max_pages} browse pages; increase --max-pages if needed")
        seen_pages.add(page_url)

        html = request_bytes(opener, page_url, timeout=timeout).decode("utf-8", "replace")
        page_downloads, page_links = crawler.discover_targets(html, page_url)
        for url in page_downloads:
            if url not in seen_downloads:
                downloads.append(url)
                seen_downloads.add(url)
        for url in page_links:
            if url not in seen_pages and url not in pending:
                pending.append(url)

    return downloads


def _copy_stream(response: BinaryIO, handle: BinaryIO, existing_bytes: int, total_bytes: int | None) -> None:
    copied = existing_bytes
    last_report = time.monotonic()
    while True:
        chunk = response.read(CHUNK_SIZE)
        if not chunk:
            break
        handle.write(chunk)
        copied += len(chunk)
        now = time.monotonic()
        if now - last_report >= 5:
            if total_bytes:
                pct = copied / total_bytes * 100
                print(f"  downloaded {copied / 1024 / 1024:.1f} MiB / {total_bytes / 1024 / 1024:.1f} MiB ({pct:.1f}%)")
            else:
                print(f"  downloaded {copied / 1024 / 1024:.1f} MiB")
            last_report = now


def download_one(
    opener: urllib.request.OpenerDirector,
    url: str,
    destination: Path,
    *,
    root_url: str,
    timeout: int,
    overwrite: bool,
) -> Path:
    output = local_path_for_url(url, destination, root_url)
    if output.exists() and output.stat().st_size > 0 and not overwrite:
        print(f"skip existing: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    existing = partial.stat().st_size if partial.exists() and not overwrite else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    with opener.open(req, timeout=timeout) as response:
        status = getattr(response, "status", None)
        mode = "ab" if existing and status == 206 else "wb"
        if mode == "wb":
            existing = 0
        content_length = response.headers.get("Content-Length")
        total = int(content_length) + existing if content_length and content_length.isdigit() else None
        print(f"download: {url}")
        with partial.open(mode) as handle:
            _copy_stream(response, handle, existing, total)

    partial.replace(output)
    print(f"saved: {output}")
    return output


def download_with_retries(
    opener: urllib.request.OpenerDirector,
    urls: list[str],
    destination: Path,
    *,
    root_url: str,
    timeout: int,
    overwrite: bool,
    retries: int,
) -> list[Path]:
    saved: list[Path] = []
    for url in urls:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                saved.append(
                    download_one(
                        opener,
                        url,
                        destination,
                        root_url=root_url,
                        timeout=timeout,
                        overwrite=overwrite,
                    )
                )
                break
            except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as exc:
                last_error = exc
                print(f"attempt {attempt}/{retries} failed for {url}: {exc}", file=sys.stderr)
                time.sleep(min(30, 2 * attempt))
        else:
            raise RuntimeError(f"could not download {url}") from last_error
    return saved


def prompt_missing(value: str | None, label: str) -> str:
    if value:
        return value
    if not sys.stdin.isatty():
        raise RuntimeError(f"missing {label}; provide it with a CLI argument or environment variable")
    if label == "email":
        return input("CIC registration email: ").strip()
    if label == "first name":
        return input("CIC registration first name: ").strip()
    if label == "last name":
        return input("CIC registration last name: ").strip()
    if label == "institution":
        return input("CIC registration institution: ").strip()
    if label == "job title":
        return input("CIC registration job title: ").strip()
    return input("CIC registration country: ").strip()


def parse_suffixes(raw: str) -> tuple[str, ...]:
    suffixes = []
    for item in raw.split(","):
        suffix = item.strip().lower()
        if not suffix:
            continue
        suffixes.append(suffix if suffix.startswith(".") else f".{suffix}")
    return tuple(suffixes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register for the official CIC IoT-DIAD 2024 browse page and download released data files."
    )
    parser.add_argument("--first-name", default=os.environ.get("CIC_FIRST_NAME"))
    parser.add_argument("--last-name", default=os.environ.get("CIC_LAST_NAME"))
    parser.add_argument("--email", default=os.environ.get("CIC_EMAIL"))
    parser.add_argument("--institution", default=os.environ.get("CIC_INSTITUTION"))
    parser.add_argument("--job-title", default=os.environ.get("CIC_JOB_TITLE", "Student"))
    parser.add_argument("--country", default=os.environ.get("CIC_COUNTRY", "China"))
    parser.add_argument("--output", default="data/raw/cic_iot_diad_2024")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--suffixes", default=",".join(DEFAULT_SUFFIXES))
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only download files whose official dataset path is under this prefix. Can be repeated.",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--insecure-tls", action="store_true", help="Disable TLS verification if a local proxy causes certificate errors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = RegistrationInfo(
        first_name=prompt_missing(args.first_name, "first name"),
        last_name=prompt_missing(args.last_name, "last name"),
        email=prompt_missing(args.email, "email"),
        institution=prompt_missing(args.institution, "institution"),
        job_title=prompt_missing(args.job_title, "job title"),
        country=prompt_missing(args.country, "country"),
    )
    if not all(info.__dict__.values()):
        raise RuntimeError("all CIC registration fields must be non-empty")

    opener = build_opener(args.proxy, args.insecure_tls)
    print("registering with CIC official form...")
    redirect = register(opener, info, args.timeout)
    start_url = urllib.parse.urljoin(ROOT_URL, f"{redirect}?t={int(time.time() * 1000)}")
    crawler = DatasetCrawler(ROOT_URL, parse_suffixes(args.suffixes), tuple(args.include_prefix))
    print(f"opening browse page: {start_url}")
    downloads = crawl_downloads(opener, start_url, crawler, max_pages=args.max_pages, timeout=args.timeout)
    if not downloads:
        raise RuntimeError("no downloadable dataset files were found on the CIC browse page")

    print(f"found {len(downloads)} downloadable file(s):")
    for url in downloads:
        print(f"- {url}")
    if args.list_only:
        return 0

    saved = download_with_retries(
        opener,
        downloads,
        Path(args.output),
        root_url=ROOT_URL,
        timeout=args.timeout,
        overwrite=args.overwrite,
        retries=args.retries,
    )
    print(f"completed {len(saved)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
