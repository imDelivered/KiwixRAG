"""ZIM file reading for RAG indexing."""

import sys
from typing import Iterator, Optional, Tuple

from kiwix_chat.config import KIWIX_BASE_URL
from kiwix_chat.kiwix.client import http_get, _auto_start_kiwix
from kiwix_chat.kiwix.parser import HTMLParserWithLinks


def list_zim_articles(zim_file_path: str) -> Iterator[Tuple[str, str]]:
    """List all articles in a ZIM file via Kiwix HTTP API.
    
    This uses the Kiwix search API to enumerate articles. Since Kiwix doesn't
    provide a direct "list all articles" endpoint, we use search with wildcards
    and common patterns to discover articles.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Yields:
        Tuples of (title, href) for each article
    """
    # Ensure Kiwix server is running
    if not _auto_start_kiwix(zim_file_path):
        print("[rag] ERROR: Could not start Kiwix server for indexing", file=sys.stderr)
        return
    
    # Kiwix doesn't have a direct "list all" API, so we use search
    # Strategy: Search for common patterns to discover articles
    # This is a limitation - we'll get a subset of articles, not all
    
    # Try searching for common single letters/patterns (Latin alphabet)
    # For non-Latin scripts, we rely on sitemap.xml which should work for all languages
    search_patterns = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    # Note: For non-Latin scripts (Chinese, Arabic, etc.), the sitemap.xml approach below
    # should discover articles. The search patterns above are just for Latin-script content.
    
    seen_hrefs = set()
    
    for pattern in search_patterns:
        try:
            from urllib.parse import quote_plus
            search_url = f"{KIWIX_BASE_URL}/search?pattern={quote_plus(pattern)}"
            html = http_get(search_url, timeout=30.0)
            
            # Parse search results
            from kiwix_chat.kiwix.parser import KiwixSearchParser
            parser = KiwixSearchParser()
            parser.feed(html)
            
            for href in parser.hrefs:
                if href not in seen_hrefs:
                    seen_hrefs.add(href)
                    # Extract title from href (approximate)
                    title = href.split('/')[-1].replace('_', ' ')
                    yield (title, href)
        
        except Exception as e:
            print(f"[rag] Warning: Error searching pattern '{pattern}': {e}", file=sys.stderr)
            continue
    
    # Also try to get articles from sitemap if available
    try:
        sitemap_url = f"{KIWIX_BASE_URL}/sitemap.xml"
        sitemap_html = http_get(sitemap_url, timeout=30.0)
        
        # Parse sitemap (simple XML parsing)
        import re
        href_pattern = r'<loc>(.*?)</loc>'
        for match in re.finditer(href_pattern, sitemap_html):
            href = match.group(1)
            # Remove base URL if present
            if href.startswith(KIWIX_BASE_URL):
                href = href[len(KIWIX_BASE_URL):]
            if href.startswith('/') and href not in seen_hrefs:
                seen_hrefs.add(href)
                title = href.split('/')[-1].replace('_', ' ')
                yield (title, href)
    except Exception:
        # Sitemap not available, that's okay
        pass


def read_zim_article(zim_file_path: str, href: str) -> Optional[str]:
    """Read article text from ZIM file via Kiwix HTTP API.
    
    Args:
        zim_file_path: Path to ZIM file (used to ensure server is running)
        href: Article href/path
        
    Returns:
        Article text content, or None if not found
    """
    # Ensure Kiwix server is running
    if not _auto_start_kiwix(zim_file_path):
        return None
    
    try:
        html = http_get(f"{KIWIX_BASE_URL}{href}", timeout=20.0)
        
        # Parse HTML to extract text
        parser = HTMLParserWithLinks()
        parser.feed(html)
        text = parser.get_text()
        
        # Clean up text
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
        return text if text else None
    except Exception as e:
        print(f"[rag] Warning: Error reading article '{href}': {e}", file=sys.stderr)
        return None


def get_article_count_estimate(zim_file_path: str) -> int:
    """Get an estimate of the number of articles in the ZIM file.
    
    This is approximate since we use search patterns.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        Estimated article count
    """
    count = 0
    for _ in list_zim_articles(zim_file_path):
        count += 1
        if count >= 1000:  # Limit enumeration for estimate
            break
    return count

