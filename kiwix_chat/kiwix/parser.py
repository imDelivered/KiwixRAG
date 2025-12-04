"""HTML parsers for Kiwix content."""

from html.parser import HTMLParser
from typing import List, Optional, Tuple

from kiwix_chat.models import ArticleLink


class HTMLParserWithLinks(HTMLParser):
    """Parse HTML to extract text and links from Kiwix articles."""
    def __init__(self):
        super().__init__()
        self.text_chunks: List[str] = []
        self.links: List[ArticleLink] = []
        self.current_link_text: List[str] = []
        self.in_link: bool = False
        self.current_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            self.in_link = True
            self.current_link_text = []
            self.current_href = None
            for attr_name, attr_value in attrs:
                if attr_name == "href" and attr_value:
                    self.current_href = attr_value

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.in_link:
            link_text = "".join(self.current_link_text).strip()
            if link_text and self.current_href:
                # Only include internal Kiwix links (start with /)
                if self.current_href.startswith("/") and not self.current_href.startswith("//"):
                    self.links.append(ArticleLink(text=link_text, href=self.current_href))
            self.in_link = False
            self.current_link_text = []
            self.current_href = None

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text_chunks.append(data)
            if self.in_link:
                self.current_link_text.append(data)

    def get_text(self) -> str:
        return "".join(self.text_chunks)

    def get_links(self) -> List[ArticleLink]:
        return self.links


class KiwixSearchParser(HTMLParser):
    """Parse Kiwix search results to extract article links."""
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []
        self.current_href: Optional[str] = None
        self.in_link: bool = False
    
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            self.in_link = True
            for attr_name, attr_value in attrs:
                if attr_name == "href" and attr_value:
                    self.current_href = attr_value
    
    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.in_link:
            if self.current_href:
                # Only include internal Kiwix links (start with /)
                if self.current_href.startswith("/") and not self.current_href.startswith("//"):
                    if self.current_href not in self.hrefs:
                        self.hrefs.append(self.current_href)
            self.in_link = False
            self.current_href = None


