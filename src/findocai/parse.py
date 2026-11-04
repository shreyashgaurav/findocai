import pymupdf

def parse_pdf(path):
    pages = []

    with pymupdf.open(path) as doc:
        for page in doc:
            pages.append({"page": page.number + 1, "text": page.get_text()})

    return pages