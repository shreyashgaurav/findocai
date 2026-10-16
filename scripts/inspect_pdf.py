"""Show what a PDF's text layer contains.

Usage:
    uv run --with pymupdf python scripts/inspect_pdf.py FILE         # every page
    uv run --with pymupdf python scripts/inspect_pdf.py FILE PAGE    # one page's text
"""

import sys

import pymupdf

doc = pymupdf.open(sys.argv[1])

if len(sys.argv) == 2:
    for page in doc:
        text_chars = len(page.get_text())
        images = len(page.get_images())
        print(f"page {page.number + 1:2}: {text_chars:5} chars, {images} images")
else:
    page = doc[int(sys.argv[2]) - 1]  # PyMuPDF counts pages from 0
    print(page.get_text())
