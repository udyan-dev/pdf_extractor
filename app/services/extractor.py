import asyncio
from collections import deque

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTContainer, LTTextLine

LAPARAMS = LAParams(all_texts=True)
_LTTextLine = LTTextLine
_LTContainer = LTContainer


def _collect_lines(root: LTContainer, lines: list[str]) -> None:
    q: deque = deque(root)
    append = lines.append
    while q:
        item = q.popleft()
        if isinstance(item, _LTTextLine):
            text = item.get_text().rstrip("\n")
            if text:
                append(text)
        elif isinstance(item, _LTContainer):
            q.extendleft(reversed(list(item)))


async def extract_pdf(path: str) -> list[dict[str, int | str]]:
    pages: list[dict[str, int | str]] = []
    append_page = pages.append
    for page_number, layout in enumerate(extract_pages(path, laparams=LAPARAMS), start=1):
        lines: list[str] = []
        _collect_lines(layout, lines)
        append_page({"page": page_number, "content": "\n".join(lines)})
        del lines
        del layout
        await asyncio.sleep(0)
    return pages
