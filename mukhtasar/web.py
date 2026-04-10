"""Web page fetching and HTML cleaning for mukhtasar."""

from __future__ import annotations

import re
from urllib.request import Request, urlopen
from urllib.error import URLError


# Tags that contain content we want to remove entirely
_STRIP_TAGS = re.compile(
    r"<\s*(script|style|nav|footer|header|aside|iframe|noscript)[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# All HTML tags
_HTML_TAGS = re.compile(r"<[^>]+>")

# HTML entities
_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&laquo;": "«", "&raquo;": "»",
}
_ENTITY_PATTERN = re.compile(r"&\w+;|&#\d+;")

# Excess whitespace
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _decode_entity(match: re.Match) -> str:
    entity = match.group(0)
    if entity in _HTML_ENTITIES:
        return _HTML_ENTITIES[entity]
    # Numeric entities like &#1575;
    if entity.startswith("&#") and entity.endswith(";"):
        try:
            return chr(int(entity[2:-1]))
        except (ValueError, OverflowError):
            return ""
    return ""


def clean_html(html: str) -> str:
    """Strip HTML to plain text. Keeps Arabic and Latin text."""
    # Remove script, style, nav, footer, etc.
    text = _STRIP_TAGS.sub("", html)

    # Convert <br>, <p>, <div>, <li> to newlines
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?\s*(p|div|li|h[1-6]|tr|blockquote)\s*[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = _HTML_TAGS.sub("", text)

    # Decode HTML entities
    text = _ENTITY_PATTERN.sub(_decode_entity, text)

    # Clean whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch a URL and return cleaned plain text.

    Uses only stdlib (no requests/beautifulsoup needed).
    """
    headers = {
        "User-Agent": "mukhtasar/0.2 (Arabic Text Summarizer)",
        "Accept": "text/html,application/xhtml+xml,text/plain",
        "Accept-Language": "ar,en;q=0.5",
    }
    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=timeout) as response:
            # Detect encoding
            content_type = response.headers.get("Content-Type", "")
            encoding = "utf-8"
            if "charset=" in content_type:
                encoding = content_type.split("charset=")[-1].split(";")[0].strip()

            raw = response.read()
            html = raw.decode(encoding, errors="replace")
    except URLError as e:
        raise ConnectionError(f"Could not fetch {url}: {e}") from e

    return clean_html(html)


def summarize_url(
    url: str,
    ratio: float = 0.3,
    max_sentences: int | None = None,
    timeout: int = 15,
):
    """Fetch a URL, extract text, and summarize it.

    Returns a Summary object just like summarize().
    """
    from mukhtasar.summarizer import summarize

    text = fetch_url(url, timeout=timeout)
    if not text:
        raise ValueError(f"No text content found at {url}")

    # Try to extract title from first line
    lines = text.strip().split("\n")
    title = None
    if lines and len(lines[0]) < 150:
        title = lines[0].strip()

    return summarize(text, ratio=ratio, max_sentences=max_sentences, title=title)
