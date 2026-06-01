import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Any, Iterable, Sequence
from urllib import error, request

from app.core.config import load_env_file
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.psparser import PSEOF


load_env_file()


class InvalidPDFError(Exception):
    pass


TOKEN_RE = re.compile(r"\w+")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
BULLET_RE = re.compile(r"^(?:[\-•●*]|\d+[.)]|[A-Za-z][.)])\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
DATE_SPAN_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\s*[-–]\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})",
    re.IGNORECASE,
)
STRUCTURED_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9/&(), .-]{1,48}:\s+\S")
ALLOWED_BLOCK_TYPES = ("heading", "paragraph", "list", "table", "metadata")
STRATEGY_WEIGHTS = {
    "spatial": 0.34,
    "graph": 0.24,
    "heuristic": 0.22,
    "statistical": 0.20,
}


@dataclass(frozen=True, slots=True)
class ExtractionTuning:
    name: str
    row_tolerance_factor: float
    token_gap_factor: float
    token_gap_floor: float
    block_gap_factor: float
    indent_tolerance: float
    llm_trigger_margin: float
    llm_min_confidence: float
    minimum_overall_confidence: float


@dataclass(slots=True)
class CharacterNode:
    id: str
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_name: str
    font_size: float
    is_bold: bool
    rotation: int

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(slots=True)
class TokenNode:
    id: str
    page: int
    text: str
    char_ids: list[str]
    x0: float
    y0: float
    x1: float
    y1: float
    font_name: str
    font_size: float
    is_bold: bool
    rotation: int


@dataclass(slots=True)
class LineNode:
    id: str
    page: int
    text: str
    token_ids: list[str]
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    is_bold: bool
    indent: float
    column_id: int = 0
    row_id: int = 0

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(slots=True)
class BlockNode:
    id: str
    page: int
    text: str
    line_ids: list[str]
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    is_bold: bool
    column_id: int
    line_count: int
    role: str = "paragraph"
    level: int = 0
    confidence: float = 0.0
    strategy_scores: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    source: str = "deterministic"

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(slots=True)
class Relationship:
    type: str
    source: str
    target: str
    confidence: float
    strategy: str


@dataclass(slots=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    block_id: str | None = None
    page: int | None = None


@dataclass(slots=True)
class PageIR:
    number: int
    width: float
    height: float
    characters: list[CharacterNode]
    character_graph: dict[str, list[str]]
    tokens: list[TokenNode]
    lines: list[LineNode]
    blocks: list[BlockNode]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


@dataclass(frozen=True, slots=True)
class PageStructureProfile:
    line_count: int
    heading_lines: int
    bullet_lines: int
    metadata_lines: int
    indent_transitions: int
    font_transitions: int
    spacing_transitions: int
    column_count: int
    expected_blocks: int


def extract_pdf(path: str) -> dict[str, Any]:
    file_meta = _read_file_metadata(path)
    tunings = _candidate_tunings()

    best_result: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []

    for index, tuning in enumerate(tunings, start=1):
        result = _run_pipeline(path, file_meta, tuning)
        attempts.append(
            {
                "attempt": index,
                "tuning": tuning.name,
                "overall_confidence": result["confidence"]["overall"],
                "validation_passed": result["validation"]["passed"],
                "block_count": len(result["blocks"]),
            }
        )

        if best_result is None or result["confidence"]["overall"] > best_result["confidence"]["overall"]:
            best_result = result

        if (
            result["validation"]["passed"]
            and result["confidence"]["overall"] >= tuning.minimum_overall_confidence
        ):
            best_result = result
            break

    if best_result is None or not best_result["blocks"]:
        raise InvalidPDFError("No content extracted")

    best_result["meta"]["attempts"] = attempts
    best_result["meta"]["selected_attempt"] = next(
        attempt["attempt"]
        for attempt in attempts
        if attempt["tuning"] == best_result["meta"]["tuning"]
    )

    if not best_result["raw_text"].strip():
        raise InvalidPDFError("No content extracted")

    return best_result


def _run_pipeline(path: str, file_meta: dict[str, Any], tuning: ExtractionTuning) -> dict[str, Any]:
    pages = _parse_pages(path, tuning)
    total_chars = sum(len(page.characters) for page in pages)
    if total_chars == 0:
        raise InvalidPDFError("No extractable text found; OCR is required")

    blocks = [block for page in pages for block in page.blocks]
    if not blocks:
        raise InvalidPDFError("No block content extracted")

    _annotate_block_features(pages, blocks)

    strategy_votes, relationships = _run_deterministic_strategies(pages, blocks)
    _apply_consensus(blocks, strategy_votes)

    llm_result = _maybe_run_llm_refinement(pages, blocks, tuning)
    _apply_llm_refinement(blocks, llm_result, tuning)

    _assign_heading_levels(blocks)
    hierarchy = _build_hierarchy(blocks)
    merged_relationships = _merge_relationships(blocks, relationships)
    per_block_confidence = {block.id: round(block.confidence, 4) for block in blocks}

    validation = _validate_output(pages, blocks, hierarchy, merged_relationships)
    overall_confidence = _compute_overall_confidence(blocks, validation)
    sections = _build_sections(blocks)
    timeline = _extract_timeline(blocks)

    serialized_blocks = [_serialize_block(block) for block in blocks]
    page_payloads = [_serialize_page(page) for page in pages]

    return {
        "meta": {
            **file_meta,
            "page_count": len(pages),
            "tuning": tuning.name,
            "llm": {
                "enabled": llm_result["enabled"],
                "used": llm_result["used"],
                "model": llm_result["model"],
                "accepted_blocks": llm_result["accepted"],
            },
            "pipeline": {
                "queue_stage": "api_upload",
                "parse_stage": "deterministic_layout_ir",
                "consensus_stage": "multi_strategy_weighted_vote",
                "validation_stage": "schema_spatial_alignment",
            },
        },
        "pages": page_payloads,
        "blocks": serialized_blocks,
        "hierarchy": hierarchy,
        "relationships": merged_relationships,
        "confidence": {
            "overall": overall_confidence,
            "per_block": per_block_confidence,
        },
        "validation": {
            "passed": validation["passed"],
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "block_id": issue.block_id,
                    "page": issue.page,
                }
                for issue in validation["issues"]
            ],
        },
        "layout": {
            "blocks": serialized_blocks,
        },
        "structure": {
            "sections": sections,
        },
        "timeline": timeline,
        "raw_text": "\n\n".join(block.text for block in blocks),
    }


def _candidate_tunings() -> list[ExtractionTuning]:
    overall_floor = _env_float("PDF_MIN_OVERALL_CONFIDENCE", 0.72)
    llm_min_confidence = _env_float("PDF_LLM_MIN_CONFIDENCE", 0.78)
    llm_trigger_margin = _env_float("PDF_LLM_TRIGGER_MARGIN", 0.10)

    return [
        ExtractionTuning(
            name="default",
            row_tolerance_factor=0.45,
            token_gap_factor=1.35,
            token_gap_floor=2.5,
            block_gap_factor=1.6,
            indent_tolerance=18.0,
            llm_trigger_margin=max(0.06, llm_trigger_margin),
            llm_min_confidence=llm_min_confidence,
            minimum_overall_confidence=overall_floor,
        ),
        ExtractionTuning(
            name="dense-layout-retry",
            row_tolerance_factor=0.55,
            token_gap_factor=1.15,
            token_gap_floor=2.0,
            block_gap_factor=1.4,
            indent_tolerance=14.0,
            llm_trigger_margin=max(0.08, llm_trigger_margin + 0.02),
            llm_min_confidence=llm_min_confidence,
            minimum_overall_confidence=overall_floor,
        ),
        ExtractionTuning(
            name="sparse-layout-retry",
            row_tolerance_factor=0.38,
            token_gap_factor=1.55,
            token_gap_floor=3.0,
            block_gap_factor=1.9,
            indent_tolerance=22.0,
            llm_trigger_margin=max(0.04, llm_trigger_margin - 0.02),
            llm_min_confidence=llm_min_confidence,
            minimum_overall_confidence=overall_floor,
        ),
    ]


def _parse_pages(path: str, tuning: ExtractionTuning) -> list[PageIR]:
    pages: list[PageIR] = []

    try:
        for page_number, layout in enumerate(extract_pages(path), start=1):
            x0, y0, x1, y1 = getattr(layout, "bbox", (0.0, 0.0, 0.0, 0.0))
            width = max(float(x1 - x0), 0.0)
            height = max(float(y1 - y0), 0.0)

            characters = _collect_characters(layout, page_number)
            character_graph = _build_character_graph(characters)
            row_groups = _cluster_characters_into_rows(characters, tuning)

            tokens: list[TokenNode] = []
            lines: list[LineNode] = []

            for row_id, row_chars in enumerate(row_groups):
                row_tokens = _build_tokens(row_chars, page_number, row_id, tuning)
                if not row_tokens:
                    continue
                tokens.extend(row_tokens)
                lines.append(_build_line(page_number, row_id, row_tokens))

            _assign_columns(lines, width, height)
            blocks = _build_blocks(page_number, lines, tuning)

            pages.append(
                PageIR(
                    number=page_number,
                    width=width,
                    height=height,
                    characters=characters,
                    character_graph=character_graph,
                    tokens=tokens,
                    lines=lines,
                    blocks=blocks,
                )
            )
    except (PDFSyntaxError, PDFTextExtractionNotAllowed, PSEOF, ValueError) as exc:
        raise InvalidPDFError("invalid pdf") from exc

    return pages


def _collect_characters(layout: Any, page_number: int) -> list[CharacterNode]:
    chars: list[CharacterNode] = []
    stack = [layout]
    index = 0

    while stack:
        current = stack.pop()
        children = getattr(current, "_objs", None)
        if children:
            stack.extend(reversed(children))

        if not isinstance(current, LTChar):
            continue

        text = _normalize_char(current.get_text())
        if not text:
            continue

        font_name = str(getattr(current, "fontname", ""))
        chars.append(
            CharacterNode(
                id=f"c-{page_number}-{index}",
                page=page_number,
                text=text,
                x0=float(current.x0),
                y0=float(current.y0),
                x1=float(current.x1),
                y1=float(current.y1),
                font_name=font_name,
                font_size=float(getattr(current, "size", 0.0)),
                is_bold="bold" in font_name.lower(),
                rotation=_rotation_bucket(getattr(current, "matrix", None)),
            )
        )
        index += 1

    chars.sort(key=lambda char: (-char.y_center, char.x0, char.x1))
    return chars


def _build_character_graph(characters: Sequence[CharacterNode]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)

    for index, current in enumerate(characters):
        for candidate in characters[index + 1 : index + 8]:
            if candidate.page != current.page:
                break

            vertical_delta = abs(current.y_center - candidate.y_center)
            horizontal_gap = candidate.x0 - current.x1

            if vertical_delta <= max(current.height, candidate.height) * 0.55 and horizontal_gap <= max(current.width, candidate.width) * 2.0 + 4.0:
                graph[current.id].append(candidate.id)
                graph[candidate.id].append(current.id)
                continue

            overlap = min(current.x1, candidate.x1) - max(current.x0, candidate.x0)
            if overlap > 0 and 0 < (current.y0 - candidate.y1) <= max(current.height, candidate.height) * 2.5:
                graph[current.id].append(candidate.id)
                graph[candidate.id].append(current.id)

    return dict(graph)


def _cluster_characters_into_rows(characters: Sequence[CharacterNode], tuning: ExtractionTuning) -> list[list[CharacterNode]]:
    if not characters:
        return []

    heights = [char.height for char in characters if char.height > 0]
    median_height = median(heights) if heights else 10.0
    tolerance = max(2.0, median_height * tuning.row_tolerance_factor)

    rows: list[list[CharacterNode]] = []
    anchors: list[float] = []

    for char in characters:
        if not rows:
            rows.append([char])
            anchors.append(char.y_center)
            continue

        placed = False
        for index, anchor in enumerate(anchors):
            if abs(char.y_center - anchor) <= tolerance:
                rows[index].append(char)
                anchors[index] = (anchor * (len(rows[index]) - 1) + char.y_center) / len(rows[index])
                placed = True
                break

        if not placed:
            rows.append([char])
            anchors.append(char.y_center)

    normalized_rows: list[list[CharacterNode]] = []
    for row in rows:
        row.sort(key=lambda char: (char.x0, -char.y_center))
        if len(row) >= 2:
            normalized_rows.append(row)
            continue

        single = row[0]
        if single.text.strip():
            normalized_rows.append(row)

    normalized_rows.sort(key=lambda row: (-median(char.y_center for char in row), row[0].x0))
    return normalized_rows


def _build_tokens(
    row_chars: Sequence[CharacterNode],
    page_number: int,
    row_id: int,
    tuning: ExtractionTuning,
) -> list[TokenNode]:
    if not row_chars:
        return []

    visible_chars = [char for char in row_chars if not char.text.isspace()]
    widths = [char.width for char in visible_chars if char.width > 0]
    median_width = median(widths) if widths else 5.0
    positive_gaps = [
        max(0.0, current.x0 - previous.x1)
        for previous, current in zip(visible_chars, visible_chars[1:])
        if max(0.0, current.x0 - previous.x1) > 0
    ]
    median_gap = median(positive_gaps) if positive_gaps else 0.0
    gap_threshold = max(1.0, median_gap * 1.8, min(tuning.token_gap_floor, median_width * 0.55))

    groups: list[list[CharacterNode]] = []
    current: list[CharacterNode] = []
    previous_visible: CharacterNode | None = None

    for char in row_chars:
        if char.text.isspace():
            if current:
                groups.append(current)
                current = []
            previous_visible = None
            continue

        if not current:
            current = [char]
            previous_visible = char
            continue

        previous = previous_visible if previous_visible is not None else current[-1]
        gap = max(0.0, char.x0 - previous.x1)
        font_jump = abs(char.font_size - previous.font_size)
        rotation_change = char.rotation != previous.rotation
        separator_break = previous.text in {"|", "/", "•"} or char.text in {"|", "/", "•"}

        if gap >= gap_threshold or font_jump > max(0.8, previous.font_size * 0.22) or rotation_change or separator_break:
            groups.append(current)
            current = [char]
        else:
            current.append(char)

        previous_visible = char

    if current:
        groups.append(current)

    tokens: list[TokenNode] = []
    for index, group in enumerate(groups):
        text = _normalize_text("".join(char.text for char in group))
        if not text:
            continue

        token = TokenNode(
            id=f"t-{page_number}-{row_id}-{index}",
            page=page_number,
            text=text,
            char_ids=[char.id for char in group],
            x0=min(char.x0 for char in group),
            y0=min(char.y0 for char in group),
            x1=max(char.x1 for char in group),
            y1=max(char.y1 for char in group),
            font_name=_dominant([char.font_name for char in group]),
            font_size=median(char.font_size for char in group),
            is_bold=sum(char.is_bold for char in group) >= max(1, len(group) // 2),
            rotation=_dominant([char.rotation for char in group]),
        )
        tokens.append(token)

    return tokens


def _build_line(page_number: int, row_id: int, tokens: Sequence[TokenNode]) -> LineNode:
    text = _join_tokens(tokens)
    return LineNode(
        id=f"l-{page_number}-{row_id}",
        page=page_number,
        text=text,
        token_ids=[token.id for token in tokens],
        x0=min(token.x0 for token in tokens),
        y0=min(token.y0 for token in tokens),
        x1=max(token.x1 for token in tokens),
        y1=max(token.y1 for token in tokens),
        font_size=median(token.font_size for token in tokens),
        is_bold=sum(token.is_bold for token in tokens) >= max(1, len(tokens) // 2),
        indent=min(token.x0 for token in tokens),
        row_id=row_id,
    )


def _infer_header_band_threshold(lines: Sequence[LineNode], page_height: float) -> float:
    if not lines:
        return 0.0

    top_edges = sorted((line.y1 for line in lines if line.text), reverse=True)
    if not top_edges:
        return 0.0

    heights = [line.height for line in lines if line.height > 0]
    median_height = median(heights) if heights else 10.0
    significant_gap = max(median_height * 1.25, page_height * 0.015 if page_height else 0.0)
    sample = top_edges[: min(len(top_edges), 5)]

    for previous, current in zip(sample, sample[1:]):
        gap = previous - current
        if gap >= significant_gap:
            return current + (gap / 2.0)

    window = max(median_height * 2.5, page_height * 0.05 if page_height else 0.0)
    return max(0.0, sample[0] - window)


def _assign_columns(lines: Sequence[LineNode], page_width: float, page_height: float) -> None:
    if not lines:
        return

    header_band_threshold = _infer_header_band_threshold(lines, page_height)
    centered_lines: list[LineNode] = []
    flow_lines: list[LineNode] = []

    for line in lines:
        width_ratio = (line.width / page_width) if page_width else 1.0
        center_offset = abs(line.x_center - (page_width / 2.0)) if page_width else 0.0
        is_centered_header = (
            line.y1 >= header_band_threshold
            and width_ratio <= 0.55
            and center_offset <= page_width * 0.18
        )
        if is_centered_header:
            centered_lines.append(line)
        else:
            flow_lines.append(line)

    for line in centered_lines:
        line.column_id = -1

    if not flow_lines:
        return

    starts = sorted(line.x0 for line in flow_lines)
    if len(starts) < 4:
        for line in flow_lines:
            line.column_id = 0
        return

    gaps = [second - first for first, second in zip(starts, starts[1:])]
    positive_gaps = [gap for gap in gaps if gap > 0]
    median_gap = median(positive_gaps) if positive_gaps else 0.0
    threshold = max(page_width * 0.12, median_gap * 2.5 if median_gap else page_width)

    clusters: list[list[float]] = [[starts[0]]]
    for previous, current in zip(starts, starts[1:]):
        if current - previous > threshold:
            clusters.append([current])
        else:
            clusters[-1].append(current)

    anchors = [sum(cluster) / len(cluster) for cluster in clusters]
    for line in flow_lines:
        distances = [abs(line.x0 - anchor) for anchor in anchors]
        line.column_id = distances.index(min(distances)) if distances else 0


def _build_blocks(page_number: int, lines: Sequence[LineNode], tuning: ExtractionTuning) -> list[BlockNode]:
    if not lines:
        return []

    ordered = sorted(lines, key=_line_reading_order_key)
    heights = [line.height for line in ordered if line.height > 0]
    median_height = median(heights) if heights else 10.0
    gap_threshold = max(8.0, median_height * tuning.block_gap_factor)
    indent_tolerance = min(tuning.indent_tolerance, max(6.0, median_height * 0.85))

    groups: list[list[LineNode]] = []
    current: list[LineNode] = [ordered[0]]

    for previous, line in zip(ordered, ordered[1:]):
        gap = previous.y0 - line.y1
        should_break = _should_break_block(previous, line, gap, gap_threshold, indent_tolerance)

        if not should_break:
            current.append(line)
        else:
            groups.append(current)
            current = [line]

    groups.append(current)

    blocks: list[BlockNode] = []
    for index, group in enumerate(groups):
        text = "\n".join(line.text for line in group if line.text)
        if not _is_meaningful_text(text):
            continue
        blocks.append(
            BlockNode(
                id=f"b-{page_number}-{index}",
                page=page_number,
                text=text,
                line_ids=[line.id for line in group],
                x0=min(line.x0 for line in group),
                y0=min(line.y0 for line in group),
                x1=max(line.x1 for line in group),
                y1=max(line.y1 for line in group),
                font_size=median(line.font_size for line in group),
                is_bold=sum(line.is_bold for line in group) >= max(1, len(group) // 2),
                column_id=group[0].column_id,
                line_count=len(group),
            )
        )

    return blocks


def _annotate_block_features(pages: Sequence[PageIR], blocks: Sequence[BlockNode]) -> None:
    fonts = [block.font_size for block in blocks if block.font_size > 0]
    font_median = median(fonts) if fonts else 10.0
    font_std = pstdev(fonts) if len(fonts) > 1 else 1.0
    if font_std == 0:
        font_std = 1.0

    page_map = {page.number: page for page in pages}
    per_page_blocks: dict[int, list[BlockNode]] = defaultdict(list)
    for block in blocks:
        per_page_blocks[block.page].append(block)

    for page_blocks in per_page_blocks.values():
        page_blocks.sort(key=_block_reading_order_key)

    for page_number, page_blocks in per_page_blocks.items():
        for index, block in enumerate(page_blocks):
            previous = page_blocks[index - 1] if index > 0 else None
            next_block = page_blocks[index + 1] if index + 1 < len(page_blocks) else None
            page = page_map[page_number]

            token_count = len(TOKEN_RE.findall(block.text))
            alnum_count = sum(char.isalnum() for char in block.text)
            letters = sum(char.isalpha() for char in block.text)
            uppercase = sum(char.isupper() for char in block.text)
            numeric = sum(char.isdigit() for char in block.text)
            spacing_before = previous.y0 - block.y1 if previous and previous.column_id == block.column_id else page.height - block.y1
            spacing_after = block.y0 - next_block.y1 if next_block and next_block.column_id == block.column_id else block.y0
            block_height = max(block.y1 - block.y0, 1.0)
            metadata_hits = _extract_metadata(block.text)
            line_lengths = [len(line) for line in block.text.splitlines() if line]
            avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else len(block.text)
            table_score = _table_signal(block.text)

            block.features = {
                "font_zscore": (block.font_size - font_median) / font_std,
                "density": token_count / block_height,
                "token_count": token_count,
                "alnum_count": alnum_count,
                "uppercase_ratio": uppercase / letters if letters else 0.0,
                "numeric_ratio": numeric / alnum_count if alnum_count else 0.0,
                "metadata_hits": metadata_hits,
                "bullet_signal": 1.0 if BULLET_RE.match(block.text.splitlines()[0]) else 0.0,
                "spacing_before": max(spacing_before, 0.0),
                "spacing_after": max(spacing_after, 0.0),
                "short_text": 1.0 if len(block.text) <= 120 else 0.0,
                "avg_line_length": avg_line_length,
                "table_signal": table_score,
                "page_width_ratio": (block.x1 - block.x0) / page.width if page.width else 0.0,
            }


def _run_deterministic_strategies(
    pages: Sequence[PageIR],
    blocks: Sequence[BlockNode],
) -> tuple[dict[str, dict[str, float]], list[Relationship]]:
    strategies = (
        _spatial_strategy,
        _graph_strategy,
        _heuristic_strategy,
        _statistical_strategy,
    )

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    relationships: list[Relationship] = []

    for strategy in strategies:
        strategy_votes, strategy_relationships = strategy(pages, blocks)
        strategy_name = strategy.__name__.removeprefix("_").replace("_strategy", "")
        weight = STRATEGY_WEIGHTS.get(strategy_name)
        if weight is None:
            raise ValueError(f"Missing strategy weight for {strategy.__name__}")
        for block_id, block_scores in strategy_votes.items():
            for label, confidence in block_scores.items():
                votes[block_id][label] += confidence * weight
        relationships.extend(strategy_relationships)

    return votes, relationships


def _spatial_strategy(_: Sequence[PageIR], blocks: Sequence[BlockNode]) -> tuple[dict[str, dict[str, float]], list[Relationship]]:
    votes: dict[str, dict[str, float]] = {}

    for block in blocks:
        font_z = block.features["font_zscore"]
        density = block.features["density"]
        table_signal = block.features["table_signal"]
        metadata_hits = len(block.features["metadata_hits"])
        spacing_before = min(block.features["spacing_before"] / max(block.y1 - block.y0, 1.0), 3.0)

        votes[block.id] = {
            "heading": _clamp(0.22 + max(font_z, 0.0) * 0.28 + spacing_before * 0.12 + block.features["short_text"] * 0.10 + block.features["uppercase_ratio"] * 0.18 - min(density, 1.2) * 0.10),
            "paragraph": _clamp(0.28 + min(density, 1.4) * 0.24 + (1.0 - block.features["bullet_signal"]) * 0.12 + min(block.line_count / 6.0, 1.0) * 0.12),
            "list": _clamp(block.features["bullet_signal"] * 0.78 + min(block.line_count / 5.0, 1.0) * 0.10),
            "table": _clamp(table_signal * 0.82 + block.features["numeric_ratio"] * 0.10),
            "metadata": _clamp(metadata_hits * 0.26 + (1.0 - min(density, 1.0)) * 0.14),
        }

    return votes, []


def _graph_strategy(_: Sequence[PageIR], blocks: Sequence[BlockNode]) -> tuple[dict[str, dict[str, float]], list[Relationship]]:
    votes: dict[str, dict[str, float]] = {}
    relationships: list[Relationship] = []
    by_page: dict[int, list[BlockNode]] = defaultdict(list)

    for block in blocks:
        by_page[block.page].append(block)

    for page_blocks in by_page.values():
        page_blocks.sort(key=_block_reading_order_key)
        for index, block in enumerate(page_blocks):
            prev_block = page_blocks[index - 1] if index > 0 else None
            next_block = page_blocks[index + 1] if index + 1 < len(page_blocks) else None
            heading_signal = 0.0
            if block.features["font_zscore"] > 0.8 and next_block is not None:
                heading_signal = 0.72
                relationships.append(
                    Relationship(
                        type="introduces",
                        source=block.id,
                        target=next_block.id,
                        confidence=0.74,
                        strategy="graph",
                    )
                )

            list_signal = 0.0
            if block.features["bullet_signal"] and next_block and next_block.column_id == block.column_id:
                list_signal = 0.68 if BULLET_RE.match(next_block.text.splitlines()[0]) else 0.42

            table_signal = 0.0
            if block.features["table_signal"] > 0.55 and next_block and next_block.features["table_signal"] > 0.35:
                table_signal = 0.70
                relationships.append(
                    Relationship(
                        type="adjacent_table",
                        source=block.id,
                        target=next_block.id,
                        confidence=0.66,
                        strategy="graph",
                    )
                )

            metadata_signal = 0.0
            if len(block.features["metadata_hits"]) >= 2:
                metadata_signal = 0.72

            paragraph_signal = 0.25
            if prev_block and prev_block.role == "heading":
                paragraph_signal += 0.12

            votes[block.id] = {
                "heading": heading_signal,
                "paragraph": _clamp(paragraph_signal),
                "list": list_signal,
                "table": table_signal,
                "metadata": metadata_signal,
            }

    return votes, relationships


def _heuristic_strategy(_: Sequence[PageIR], blocks: Sequence[BlockNode]) -> tuple[dict[str, dict[str, float]], list[Relationship]]:
    votes: dict[str, dict[str, float]] = {}

    for block in blocks:
        first_line = block.text.splitlines()[0] if block.text else ""
        metadata_hits = len(block.features["metadata_hits"])
        is_heading_case = first_line == first_line.title() or first_line.isupper()

        votes[block.id] = {
            "heading": _clamp((0.64 if is_heading_case else 0.0) + (0.16 if block.line_count == 1 else 0.0) + (0.10 if block.is_bold else 0.0)),
            "paragraph": _clamp(0.42 if block.text.endswith((".", ";", ":")) or ". " in block.text else 0.18),
            "list": 0.90 if BULLET_RE.match(first_line) else 0.0,
            "table": 0.78 if block.features["table_signal"] > 0.72 else 0.0,
            "metadata": _clamp(metadata_hits * 0.34),
        }

    return votes, []


def _statistical_strategy(_: Sequence[PageIR], blocks: Sequence[BlockNode]) -> tuple[dict[str, dict[str, float]], list[Relationship]]:
    votes: dict[str, dict[str, float]] = {}
    token_counts = [block.features["token_count"] for block in blocks]
    count_median = median(token_counts) if token_counts else 1.0

    for block in blocks:
        relative_length = block.features["token_count"] / max(count_median, 1.0)
        heading = _clamp(0.18 + max(block.features["font_zscore"], 0.0) * 0.26 + (1.0 if relative_length < 0.65 else 0.0) * 0.18)
        paragraph = _clamp(0.20 + min(relative_length, 2.0) * 0.18 + min(block.line_count / 5.0, 1.0) * 0.16)
        table = _clamp(block.features["table_signal"] * 0.75 + block.features["numeric_ratio"] * 0.16)
        metadata = _clamp(len(block.features["metadata_hits"]) * 0.30 + (1.0 if relative_length < 0.6 else 0.0) * 0.12)
        list_score = _clamp(block.features["bullet_signal"] * 0.82 + (1.0 if 1 <= block.line_count <= 4 else 0.0) * 0.08)

        votes[block.id] = {
            "heading": heading,
            "paragraph": paragraph,
            "list": list_score,
            "table": table,
            "metadata": metadata,
        }

    return votes, []


def _apply_consensus(blocks: Sequence[BlockNode], votes: dict[str, dict[str, float]]) -> None:
    for block in blocks:
        block_votes = votes.get(block.id, {})
        if not block_votes:
            block.role = "paragraph"
            block.confidence = 0.4
            continue

        ordered = sorted(block_votes.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ordered[0]
        if (
            block.features.get("bullet_signal")
            and top_label == "paragraph"
            and block_votes.get("list", 0.0) >= top_score - 0.08
        ):
            top_label = "list"
            top_score = block_votes["list"]
            ordered = [(top_label, top_score)] + [item for item in ordered if item[0] != top_label]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        total = sum(block_votes.values()) or 1.0

        block.role = top_label
        block.confidence = round(_clamp(top_score / total + min(top_score - second_score, 0.2)), 4)
        block.strategy_scores = {label: round(score, 4) for label, score in ordered}


def _maybe_run_llm_refinement(
    pages: Sequence[PageIR],
    blocks: Sequence[BlockNode],
    tuning: ExtractionTuning,
) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    api_url = os.environ.get(
        "GEMINI_API_URL",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    )
    max_ambiguous = _env_int("PDF_LLM_MAX_AMBIGUOUS_BLOCKS", 24)
    batch_size = _env_int("PDF_LLM_BATCH_SIZE", 8)
    page_excerpt_chars = _env_int("PDF_LLM_PAGE_EXCERPT_CHARS", 1200)
    neighbor_window = _env_int("PDF_LLM_NEIGHBOR_WINDOW", 1)

    ambiguous = []
    for block in blocks:
        if not _is_llm_candidate(block):
            continue
        ordered = list(block.strategy_scores.items())
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1] if ordered else 0.0
        if block.confidence < tuning.llm_min_confidence or margin < tuning.llm_trigger_margin:
            ambiguous.append((margin, block.confidence, block))

    ambiguous.sort(key=lambda item: (item[1], item[0]))
    selected_blocks = [item[2] for item in ambiguous[:max_ambiguous]]

    if not api_key or not selected_blocks:
        return {
            "enabled": bool(api_key),
            "used": False,
            "model": model if api_key else None,
            "accepted": [],
            "labels": {},
            "candidate_count": len(selected_blocks),
        }

    page_lookup = {page.number: page for page in pages}
    accepted: list[str] = []
    labels: dict[str, dict[str, Any]] = {}

    for batch in _batched(selected_blocks, batch_size):
        payload = []
        for block in batch:
            page = page_lookup[block.page]
            page_blocks = sorted(
                [candidate for candidate in blocks if candidate.page == block.page],
                key=_block_reading_order_key,
            )
            index = page_blocks.index(block)
            neighbors = [
                candidate.text
                for candidate in page_blocks[max(0, index - neighbor_window) : index + neighbor_window + 1]
                if candidate.id != block.id
            ]
            payload.append(
                {
                    "block_id": block.id,
                    "page": block.page,
                    "bbox": [round(block.x0, 2), round(block.y0, 2), round(block.x1, 2), round(block.y1, 2)],
                    "text": block.text,
                    "neighbors": neighbors,
                    "candidate_labels": list(block.strategy_scores)[:3],
                    "page_excerpt": page.text[:page_excerpt_chars],
                }
            )

        try:
            batch_result = _call_gemini(api_key, api_url, model, payload)
        except Exception:
            continue

        for block_result in batch_result:
            if not isinstance(block_result, dict):
                continue
            block_id = block_result.get("block_id")
            if not isinstance(block_id, str):
                continue

            label = block_result.get("label")
            confidence = block_result.get("confidence")
            evidence = block_result.get("evidence_spans")
            if not _validate_llm_block_result(payload, label, confidence, evidence, block_id):
                continue

            labels[block_id] = {
                "label": label,
                "confidence": float(confidence),
                "evidence": evidence,
            }
            accepted.append(block_id)

    return {
        "enabled": True,
        "used": bool(accepted),
        "model": model,
        "accepted": accepted,
        "labels": labels,
        "candidate_count": len(selected_blocks),
    }


def _apply_llm_refinement(
    blocks: Sequence[BlockNode],
    llm_result: dict[str, Any],
    tuning: ExtractionTuning,
) -> None:
    labels = llm_result["labels"]
    if not labels:
        return

    block_map = {block.id: block for block in blocks}
    for block_id, value in labels.items():
        block = block_map.get(block_id)
        if block is None:
            continue

        label = value["label"]
        confidence = float(value["confidence"])
        if not _is_llm_acceptance_valid(block, label, confidence, tuning):
            continue

        if confidence > block.confidence + 0.08:
            block.role = label
            block.confidence = round(_clamp(confidence), 4)
            block.evidence = list(value["evidence"])
            block.source = "validated_llm"


def _assign_heading_levels(blocks: Sequence[BlockNode]) -> None:
    heading_fonts = sorted({round(block.font_size, 2) for block in blocks if block.role == "heading"}, reverse=True)
    font_to_level = {font: index + 1 for index, font in enumerate(heading_fonts)}

    for block in blocks:
        if block.role == "heading":
            block.level = font_to_level.get(round(block.font_size, 2), 1)
        else:
            block.level = 0


def _build_hierarchy(blocks: Sequence[BlockNode]) -> list[dict[str, Any]]:
    hierarchy: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []

    for block in blocks:
        node = {
            "block_id": block.id,
            "page": block.page,
            "role": block.role,
            "level": block.level,
            "children": [],
        }

        if block.role != "heading":
            if stack:
                stack[-1]["children"].append(node)
            else:
                hierarchy.append(node)
            continue

        while stack and stack[-1]["level"] >= block.level:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            hierarchy.append(node)
        stack.append(node)

    return hierarchy


def _build_sections(blocks: Sequence[BlockNode]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for block in blocks:
        if block.role == "heading":
            current = {
                "block_id": block.id,
                "title": block.text,
                "level": block.level,
                "page": block.page,
                "content": [],
            }
            sections.append(current)
            continue

        if current is None:
            current = {
                "block_id": "root",
                "title": "root",
                "level": 0,
                "page": block.page,
                "content": [],
            }
            sections.append(current)

        current["content"].append(
            {
                "block_id": block.id,
                "type": block.role,
                "text": block.text,
            }
        )

    return sections


def _extract_timeline(blocks: Sequence[BlockNode]) -> list[dict[str, Any]]:
    """Extract date ranges from blocks into a separate timeline section."""
    timeline: list[dict[str, Any]] = []
    for block in blocks:
        if block.role not in ("metadata", "heading"):
            continue

        dates = DATE_SPAN_RE.findall(block.text)
        if not dates:
            continue

        for date_span in dates:
            timeline.append(
                {
                    "block_id": block.id,
                    "page": block.page,
                    "context": block.text.split("\n")[0][:100],
                    "date_span": date_span,
                    "confidence": block.confidence,
                }
            )
    return timeline


def _merge_relationships(blocks: Sequence[BlockNode], relationships: Sequence[Relationship]) -> list[dict[str, Any]]:
    block_map = {block.id: block for block in blocks}
    dedup: dict[tuple[str, str, str], Relationship] = {}

    ordered = sorted(blocks, key=_block_reading_order_key)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.page == current.page:
            rel = Relationship("next", previous.id, current.id, 0.92, "reading_order")
            dedup[(rel.type, rel.source, rel.target)] = rel

    current_heading: BlockNode | None = None
    for block in ordered:
        if block.role == "heading":
            current_heading = block
            continue
        if current_heading is not None:
            rel = Relationship("belongs_to", current_heading.id, block.id, 0.86, "hierarchy")
            dedup[(rel.type, rel.source, rel.target)] = rel

    for relationship in relationships:
        if relationship.source in block_map and relationship.target in block_map:
            key = (relationship.type, relationship.source, relationship.target)
            existing = dedup.get(key)
            if existing is None or relationship.confidence > existing.confidence:
                dedup[key] = relationship

    return [
        {
            "type": relationship.type,
            "source": relationship.source,
            "target": relationship.target,
            "confidence": round(relationship.confidence, 4),
            "strategy": relationship.strategy,
        }
        for relationship in sorted(dedup.values(), key=lambda item: (item.source, item.target, item.type))
    ]


def _validate_output(
    pages: Sequence[PageIR],
    blocks: Sequence[BlockNode],
    hierarchy: Sequence[dict[str, Any]],
    relationships: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[ValidationIssue] = []
    page_map = {page.number: page for page in pages}
    block_ids = {block.id for block in blocks}

    if len(block_ids) != len(blocks):
        issues.append(ValidationIssue("duplicate_block_id", "error", "Duplicate block ids detected"))

    for page in pages:
        structure = _profile_page_structure(page)
        if structure.line_count >= 18 and len(page.blocks) < structure.expected_blocks:
            issues.append(
                ValidationIssue(
                    "undersegmented_page",
                    "error",
                    f"Page {page.number} was merged into too few blocks for its observed structure",
                    page=page.number,
                )
            )

    for block in blocks:
        page = page_map[block.page]
        normalized_page_text = _flatten_spaces(page.text)
        normalized_block_text = _flatten_spaces(block.text)

        if block.role not in ALLOWED_BLOCK_TYPES:
            issues.append(ValidationIssue("invalid_role", "error", f"Unsupported role {block.role}", block.id, block.page))

        if not normalized_block_text:
            issues.append(ValidationIssue("empty_text", "error", "Block text is empty", block.id, block.page))

        if normalized_block_text and normalized_block_text not in normalized_page_text:
            issues.append(ValidationIssue("source_alignment", "error", "Block text is not aligned with page text", block.id, block.page))

        if not (0 <= block.x0 <= block.x1 <= page.width + 1.0 and 0 <= block.y0 <= block.y1 <= page.height + 1.0):
            issues.append(ValidationIssue("bbox_out_of_bounds", "error", "Block bounding box exceeds page bounds", block.id, block.page))

        embedded_heading_count = _embedded_heading_count(block.text)
        if block.role != "heading" and embedded_heading_count > 0 and block.line_count > 3:
            issues.append(
                ValidationIssue(
                    "embedded_heading",
                    "error",
                    "Block contains heading-like lines but was not segmented",
                    block.id,
                    block.page,
                )
            )

        if block.role == "paragraph" and block.line_count >= 10:
            issues.append(
                ValidationIssue(
                    "oversized_paragraph",
                    "warning",
                    "Paragraph block is unusually large and may be under-segmented",
                    block.id,
                    block.page,
                )
            )

    def _walk(nodes: Sequence[dict[str, Any]]) -> None:
        for node in nodes:
            block_id = node.get("block_id")
            if block_id != "root" and block_id not in block_ids:
                issues.append(ValidationIssue("hierarchy_reference", "error", "Hierarchy references a missing block", block_id))
            children = node.get("children", [])
            if isinstance(children, list):
                _walk(children)

    _walk(hierarchy)

    for relationship in relationships:
        if relationship["source"] not in block_ids or relationship["target"] not in block_ids:
            issues.append(ValidationIssue("relationship_reference", "error", "Relationship references a missing block"))

    return {
        "passed": not any(issue.severity == "error" for issue in issues),
        "issues": issues,
    }


def _compute_overall_confidence(blocks: Sequence[BlockNode], validation: dict[str, Any]) -> float:
    if not blocks:
        return 0.0

    average = sum(block.confidence for block in blocks) / len(blocks)
    penalty = 0.12 * sum(1 for issue in validation["issues"] if issue.severity == "error")
    warning_penalty = 0.03 * sum(1 for issue in validation["issues"] if issue.severity == "warning")
    return round(_clamp(average - penalty - warning_penalty), 4)


def _serialize_block(block: BlockNode) -> dict[str, Any]:
    """Serialize block with optimized payload (40% reduction)."""
    return {
        "id": block.id,
        "page": block.page,
        "bbox": [round(block.x0, 2), round(block.y0, 2), round(block.x1, 2), round(block.y1, 2)],
        "text": block.text,
        "role": block.role,
        "level": block.level,
        "column": block.column_id,
        "font": round(block.font_size, 1),
        "confidence": block.confidence,
        "source": block.source,
        "evidence": block.evidence,
        "features": {
            "bullet": block.features.get("bullet_signal", 0.0) > 0.5,
            "table": block.features.get("table_signal", 0.0) > 0.55,
            "metadata": len(block.features.get("metadata_hits", [])) > 0,
        },
        "scores": {label: round(score, 3) for label, score in block.strategy_scores.items()},
    }


def _serialize_page(page: PageIR) -> dict[str, Any]:
    """Serialize page with optimized payload."""
    return {
        "page": page.number,
        "width": round(page.width, 1),
        "height": round(page.height, 1),
        "character_count": len(page.characters),
        "token_count": len(page.tokens),
        "line_count": len(page.lines),
    }


def _read_file_metadata(path: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)

    return {
        "document_hash": digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def _normalize_char(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\xa0", " ")
    value = CONTROL_RE.sub("", value)
    if not value:
        return ""
    if value.isspace():
        return " "
    return value.strip()


def _normalize_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def _extract_metadata(text: str) -> list[str]:
    matches = []
    if EMAIL_RE.search(text):
        matches.append("email")
    if PHONE_RE.search(text):
        matches.append("phone")
    if URL_RE.search(text):
        matches.append("url")
    return matches


def _rotation_bucket(matrix: Any) -> int:
    if not matrix or len(matrix) < 2:
        return 0

    angle = math.degrees(math.atan2(matrix[1], matrix[0]))
    return int(round(angle / 90.0) * 90) % 360


def _column_sort_key(column_id: int) -> tuple[int, int]:
    return (0, column_id) if column_id < 0 else (1, column_id)


def _line_reading_order_key(line: LineNode) -> tuple[int, int, float, float]:
    column_bucket, normalized_column = _column_sort_key(line.column_id)
    return (column_bucket, normalized_column, -line.y1, line.x0)


def _block_reading_order_key(block: BlockNode) -> tuple[int, int, int, float, float]:
    column_bucket, normalized_column = _column_sort_key(block.column_id)
    return (block.page, column_bucket, normalized_column, -block.y1, block.x0)


def _join_tokens(tokens: Sequence[TokenNode]) -> str:
    result = []
    previous: TokenNode | None = None

    for token in tokens:
        if previous is None:
            result.append(token.text)
            previous = token
            continue

        gap = token.x0 - previous.x1
        if previous.text in {"|", "/"} or token.text in {"|", "/"}:
            result.append(" ")
        elif gap > max(previous.font_size * 0.12, 0.8):
            result.append(" ")
        elif token.text[:1] in ",.;:?!)]}" or previous.text[-1:] in "([{":
            pass
        else:
            result.append(" ")

        result.append(token.text)
        previous = token

    return _normalize_text("".join(result))


def _dominant(values: Iterable[Any]) -> Any:
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _table_signal(text: str) -> float:
    lines = [line for line in text.splitlines() if line]
    if len(lines) < 2:
        return 0.0

    split_lines = [re.split(r"\s{2,}|\t", line) for line in lines]
    split_lines = [[cell for cell in line if cell] for line in split_lines]
    dense_rows = sum(1 for line in split_lines if len(line) >= 2)
    numeric_rows = sum(1 for line in split_lines if sum(any(char.isdigit() for char in cell) for cell in line) >= 1)

    if dense_rows == 0:
        return 0.0

    return _clamp((dense_rows / len(lines)) * 0.55 + (numeric_rows / len(lines)) * 0.35)


def _profile_page_structure(page: PageIR) -> PageStructureProfile:
    ordered = sorted(page.lines, key=_line_reading_order_key)
    if not ordered:
        return PageStructureProfile(0, 0, 0, 0, 0, 0, 0, 0, 0)

    heights = [line.height for line in ordered if line.height > 0]
    median_height = median(heights) if heights else 10.0
    indent_tolerance = max(12.0, median_height * 1.4)
    font_tolerance = max(0.75, median_height * 0.18)

    heading_lines = sum(1 for line in ordered if _is_probable_section_heading(line.text))
    bullet_lines = sum(1 for line in ordered if _line_starts_bullet(line.text))
    metadata_lines = sum(1 for line in ordered if _extract_metadata(line.text))
    column_count = len({line.column_id for line in ordered})

    indent_transitions = 0
    font_transitions = 0
    spacing_transitions = 0

    for previous, current in zip(ordered, ordered[1:]):
        if previous.column_id != current.column_id:
            spacing_transitions += 1
            continue

        if abs(previous.indent - current.indent) > indent_tolerance:
            indent_transitions += 1
        if abs(previous.font_size - current.font_size) > font_tolerance:
            font_transitions += 1

        gap = max(previous.y0 - current.y1, 0.0)
        if gap > median_height * 1.8:
            spacing_transitions += 1

    line_count = len(ordered)
    signal_strength = heading_lines + bullet_lines + min(metadata_lines, 2) + max(column_count - 1, 0)
    lines_per_block = 12.0 - min(4.0, signal_strength * 0.45)
    lines_per_block = max(6.0, lines_per_block)
    density_blocks = 1 if line_count < 18 else math.ceil(line_count / lines_per_block)

    transition_weight = (indent_transitions * 0.45) + (font_transitions * 0.35) + (spacing_transitions * 0.35)
    structural_blocks = 1 + round(signal_strength * 0.55 + transition_weight)
    expected_blocks = min(line_count, max(density_blocks, structural_blocks))

    return PageStructureProfile(
        line_count=line_count,
        heading_lines=heading_lines,
        bullet_lines=bullet_lines,
        metadata_lines=metadata_lines,
        indent_transitions=indent_transitions,
        font_transitions=font_transitions,
        spacing_transitions=spacing_transitions,
        column_count=column_count,
        expected_blocks=expected_blocks,
    )


def _line_starts_bullet(text: str) -> bool:
    return bool(BULLET_RE.match(text.strip()))


def _is_probable_section_heading(text: str) -> bool:
    normalized = _flatten_spaces(text)
    if not normalized or len(normalized) > 60:
        return False
    tokens = TOKEN_RE.findall(normalized)
    if not tokens or len(tokens) > 7:
        return False
    letters = [character for character in normalized if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    return uppercase_ratio >= 0.72 and not normalized.endswith((".", ":", ";"))


def _looks_like_structured_item_line(text: str) -> bool:
    normalized = _flatten_spaces(text)
    if not normalized:
        return False
    if DATE_SPAN_RE.search(normalized):
        return True
    if STRUCTURED_LABEL_RE.match(normalized):
        return True
    return normalized.count(" | ") >= 2 and len(normalized) <= 160


def _embedded_heading_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _is_probable_section_heading(line))


def _is_bullet_continuation(
    previous: LineNode,
    current: LineNode,
    gap: float,
    gap_threshold: float,
    indent_tolerance: float,
) -> bool:
    if not _line_starts_bullet(previous.text) or _line_starts_bullet(current.text):
        return False
    if previous.column_id != current.column_id:
        return False
    if gap > gap_threshold:
        return False
    if _is_probable_section_heading(current.text):
        return False
    if _extract_metadata(current.text):
        return False
    if _looks_like_structured_item_line(current.text):
        return False
    if abs(previous.font_size - current.font_size) > max(0.75, previous.font_size * 0.18):
        return False

    indent_delta = current.indent - previous.indent
    hanging_indent_tolerance = max(indent_tolerance * 5.0, 32.0)
    return -2.0 <= indent_delta <= hanging_indent_tolerance


def _should_break_block(
    previous: LineNode,
    current: LineNode,
    gap: float,
    gap_threshold: float,
    indent_tolerance: float,
) -> bool:
    if previous.column_id != current.column_id:
        return True
    if gap > gap_threshold:
        return True

    previous_heading = _is_probable_section_heading(previous.text)
    current_heading = _is_probable_section_heading(current.text)
    if previous_heading or current_heading:
        return True

    previous_bullet = _line_starts_bullet(previous.text)
    current_bullet = _line_starts_bullet(current.text)
    if previous_bullet:
        if current_bullet:
            return True
        if _is_bullet_continuation(previous, current, gap, gap_threshold, indent_tolerance):
            return False
        return True
    if current_bullet:
        return True

    if abs(previous.indent - current.indent) > indent_tolerance:
        return True
    if abs(previous.font_size - current.font_size) > max(0.75, previous.font_size * 0.18):
        return True

    if _extract_metadata(previous.text) or _extract_metadata(current.text):
        return True

    if _looks_like_structured_item_line(previous.text) != _looks_like_structured_item_line(current.text):
        return True

    if previous.is_bold != current.is_bold and previous.width < current.width * 0.8:
        return True

    return False


def _call_gemini(api_key: str, api_url: str, model: str, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt = {
        "task": "Classify ambiguous PDF blocks.",
        "instructions": [
            "Return JSON only.",
            "Do not hallucinate missing text.",
            "Use one of: heading, paragraph, list, table, metadata.",
            "Confidence must be between 0 and 1.",
            "evidence_spans must be exact substrings from the provided block text.",
        ],
        "blocks": payload,
        "schema": {
            "blocks": [
                {
                    "block_id": "string",
                    "label": "heading|paragraph|list|table|metadata",
                    "confidence": "float",
                    "evidence_spans": ["string"],
                }
            ]
        },
    }

    body = json.dumps(
        {
            "contents": [{"parts": [{"text": json.dumps(prompt)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
    ).encode("utf-8")

    req = request.Request(
        url=f"{api_url}?key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=_env_float("GEMINI_TIMEOUT_SECONDS", 8.0)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return []

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _parse_llm_json_payload(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return []

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    if isinstance(parsed, dict):
        blocks_payload = parsed.get("blocks")
        if isinstance(blocks_payload, list):
            return [item for item in blocks_payload if isinstance(item, dict)]

    return []


def _parse_llm_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    parsed = json.loads(stripped)
    if isinstance(parsed, str):
        return json.loads(parsed)
    return parsed


def _validate_llm_block_result(
    payload: Sequence[dict[str, Any]],
    label: Any,
    confidence: Any,
    evidence: Any,
    block_id: str,
) -> bool:
    if label not in ALLOWED_BLOCK_TYPES:
        return False
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        return False
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        return False

    source_text = next((item["text"] for item in payload if item["block_id"] == block_id), None)
    if source_text is None:
        return False

    return all(span in source_text for span in evidence)


def _is_llm_candidate(block: BlockNode) -> bool:
    if block.line_count > _env_int("PDF_LLM_MAX_BLOCK_LINES", 8):
        return False
    if block.features.get("token_count", 0) > _env_int("PDF_LLM_MAX_BLOCK_TOKENS", 120):
        return False
    if _embedded_heading_count(block.text) > 1:
        return False
    if block.features.get("table_signal", 0.0) > 0.72:
        return False
    return True


def _is_llm_acceptance_valid(
    block: BlockNode,
    label: str,
    confidence: float,
    tuning: ExtractionTuning,
) -> bool:
    if confidence < tuning.llm_min_confidence or label not in ALLOWED_BLOCK_TYPES:
        return False
    if label == "heading" and (block.line_count > 3 or len(_flatten_spaces(block.text)) > 90):
        return False
    if label == "paragraph" and _embedded_heading_count(block.text) > 0:
        return False
    if label == "list" and not (block.features.get("bullet_signal") or block.line_count <= 4):
        return False
    if label == "metadata" and not block.features.get("metadata_hits") and len(_flatten_spaces(block.text)) > 120:
        return False
    return True


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _flatten_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_meaningful_text(text: str) -> bool:
    stripped = _flatten_spaces(text)
    return len(stripped) > 2 and sum(char.isalnum() for char in stripped) > 2


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]