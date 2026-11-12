import re
import warnings

import pymupdf
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Filings are XHTML; the forgiving HTML parser reads them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PAGE_MARKER = "<<<PAGE_BREAK>>>"


def parse_pdf(path):
    """Return one dict per page: its page number and its text."""
    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            pages.append({"page": page.number + 1, "text": page.get_text()})
    return pages


def parse_filing(path):
    """Split an EDGAR filing's HTML into pages at its page-break markers."""
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Remove hidden content; readers never see it.
    for hidden in soup.find_all(style=re.compile(r"display:\s*none")):
        hidden.decompose()

    # Remove the inline-XBRL header (machine-readable data, not page text).
    for header in soup.find_all("ix:header"):
        header.decompose()

    # Put a marker after every page break, so we can split the text there.
    for element in soup.find_all(style=re.compile(r"page-break-after:\s*always")):
        element.insert_after(PAGE_MARKER)

    # Line breaks only where the layout breaks; " | " between table cells.
    for block in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4"]):
        block.append("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.append(" | ")
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.body.get_text()
    pages = []
    for number, page_text in enumerate(text.split(PAGE_MARKER), start=1):
        page_text = re.sub(r"[ \t\xa0]+", " ", page_text)  # collapse spaces
        lines = []
        for line in page_text.split("\n"):
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                lines.append(" | ".join(cells))
        page_text = "\n".join(lines)
        pages.append({"page": number, "text": page_text})
    return pages
