import unittest

from app.services.extractor import (
    BlockNode,
    CharacterNode,
    ExtractionTuning,
    LineNode,
    PageIR,
    _annotate_block_features,
    _apply_consensus,
    _assign_columns,
    _block_reading_order_key,
    _build_blocks,
    _build_line,
    _build_tokens,
    _compute_overall_confidence,
    _looks_like_structured_item_line,
    _run_deterministic_strategies,
    _validate_output,
)


TEST_TUNING = ExtractionTuning(
    name="test",
    row_tolerance_factor=0.45,
    token_gap_factor=1.35,
    token_gap_floor=2.5,
    block_gap_factor=1.6,
    indent_tolerance=18.0,
    llm_trigger_margin=0.08,
    llm_min_confidence=0.78,
    minimum_overall_confidence=0.72,
)


def _glyph(index: int, text: str, x0: float, x1: float, y0: float = 700.0, y1: float = 710.0) -> CharacterNode:
    return CharacterNode(
        id=f"c{index}",
        page=1,
        text=text,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        font_name="Test",
        font_size=10.5,
        is_bold=False,
        rotation=0,
    )


class ResumeExtractorTests(unittest.TestCase):
    def test_centered_header_orders_before_body(self) -> None:
        lines = [
            LineNode(id="name", page=1, text="Udyan Singh", token_ids=["t1"], x0=260.0, y0=734.0, x1=350.0, y1=750.0, font_size=16.0, is_bold=True, indent=260.0),
            LineNode(id="contact", page=1, text="Ahmedabad | udyan.dev@gmail.com | Github", token_ids=["t2"], x0=124.0, y0=721.0, x1=488.0, y1=732.0, font_size=10.5, is_bold=False, indent=124.0),
            LineNode(id="body", page=1, text="WORK EXPERIENCE", token_ids=["t3"], x0=44.0, y0=700.0, x1=190.0, y1=710.0, font_size=10.5, is_bold=True, indent=44.0),
        ]

        _assign_columns(lines, 612.0, 792.0)
        blocks = [
            BlockNode(id=line.id, page=1, text=line.text, line_ids=[line.id], x0=line.x0, y0=line.y0, x1=line.x1, y1=line.y1, font_size=line.font_size, is_bold=line.is_bold, column_id=line.column_id, line_count=1)
            for line in lines
        ]

        ordered_ids = [block.id for block in sorted(blocks, key=_block_reading_order_key)]

        self.assertEqual(lines[0].column_id, -1)
        self.assertEqual(ordered_ids, ["name", "contact", "body"])

    def test_centered_header_uses_content_relative_band(self) -> None:
        lines = [
            LineNode(id="name", page=1, text="Udyan Singh", token_ids=["t1"], x0=260.0, y0=574.0, x1=350.0, y1=590.0, font_size=16.0, is_bold=True, indent=260.0),
            LineNode(id="contact", page=1, text="Ahmedabad | udyan.dev@gmail.com | Github", token_ids=["t2"], x0=124.0, y0=561.0, x1=488.0, y1=572.0, font_size=10.5, is_bold=False, indent=124.0),
            LineNode(id="body", page=1, text="WORK EXPERIENCE", token_ids=["t3"], x0=44.0, y0=540.0, x1=190.0, y1=550.0, font_size=10.5, is_bold=True, indent=44.0),
        ]

        _assign_columns(lines, 612.0, 792.0)
        blocks = [
            BlockNode(id=line.id, page=1, text=line.text, line_ids=[line.id], x0=line.x0, y0=line.y0, x1=line.x1, y1=line.y1, font_size=line.font_size, is_bold=line.is_bold, column_id=line.column_id, line_count=1)
            for line in lines
        ]

        ordered_ids = [block.id for block in sorted(blocks, key=_block_reading_order_key)]

        self.assertEqual(lines[0].column_id, -1)
        self.assertEqual(ordered_ids, ["name", "contact", "body"])

    def test_build_tokens_restores_spaces_from_gaps(self) -> None:
        row = [
            _glyph(1, "W", 10.0, 16.0),
            _glyph(2, "o", 16.1, 21.0),
            _glyph(3, "r", 21.1, 25.0),
            _glyph(4, "k", 25.1, 30.0),
            _glyph(5, "E", 33.6, 39.5),
            _glyph(6, "x", 39.6, 44.5),
            _glyph(7, "p", 44.6, 49.8),
            _glyph(8, "e", 49.9, 54.8),
            _glyph(9, "r", 54.9, 58.7),
            _glyph(10, "i", 58.8, 60.8),
            _glyph(11, "e", 60.9, 65.8),
            _glyph(12, "n", 65.9, 71.2),
            _glyph(13, "c", 71.3, 76.0),
            _glyph(14, "e", 76.1, 81.0),
        ]

        tokens = _build_tokens(row, 1, 0, TEST_TUNING)
        line = _build_line(1, 0, tokens)

        self.assertEqual(line.text, "Work Experience")

    def test_build_blocks_splits_resume_sections(self) -> None:
        lines = [
            LineNode(id="name", page=1, text="Udyan Singh", token_ids=["t1"], x0=260.0, y0=734.0, x1=350.0, y1=750.0, font_size=16.0, is_bold=True, indent=260.0),
            LineNode(id="contact", page=1, text="Ahmedabad | udyan.dev@gmail.com | Github", token_ids=["t2"], x0=124.0, y0=721.0, x1=488.0, y1=732.0, font_size=10.5, is_bold=False, indent=124.0),
            LineNode(id="section", page=1, text="WORK EXPERIENCE", token_ids=["t3"], x0=44.0, y0=700.0, x1=190.0, y1=710.0, font_size=10.5, is_bold=True, indent=44.0),
            LineNode(id="role", page=1, text="Junior Software Engineer Aug 2024 - Present", token_ids=["t4"], x0=44.0, y0=682.0, x1=340.0, y1=692.0, font_size=10.5, is_bold=False, indent=44.0),
            LineNode(id="bullet", page=1, text="● Built scalable mobile modules", token_ids=["t5"], x0=60.0, y0=664.0, x1=320.0, y1=674.0, font_size=10.5, is_bold=False, indent=60.0),
        ]

        _assign_columns(lines, 612.0, 792.0)
        blocks = _build_blocks(1, lines, TEST_TUNING)

        self.assertEqual(len(blocks), 5)
        self.assertEqual([block.text.split("\n")[0] for block in blocks], [
            "Udyan Singh",
            "Ahmedabad | udyan.dev@gmail.com | Github",
            "WORK EXPERIENCE",
            "Junior Software Engineer Aug 2024 - Present",
            "● Built scalable mobile modules",
        ])

    def test_validate_output_flags_undersegmented_resume_page(self) -> None:
        lines = [
            LineNode(id=f"l{i}", page=1, text=f"Line {i}", token_ids=[f"t{i}"], x0=40.0, y0=700.0 - i, x1=500.0, y1=710.0 - i, font_size=10.5, is_bold=False, indent=40.0)
            for i in range(44)
        ]
        page = PageIR(number=1, width=612.0, height=792.0, characters=[], character_graph={}, tokens=[], lines=lines, blocks=[])
        block = BlockNode(
            id="b-1-0",
            page=1,
            text="WORK EXPERIENCE\nRAPIDOPS\nJunior Software Engineer\nEDUCATION\nBachelor of Engineering",
            line_ids=["l1", "l2", "l3", "l4", "l5"],
            x0=40.0,
            y0=100.0,
            x1=500.0,
            y1=700.0,
            font_size=10.5,
            is_bold=False,
            column_id=0,
            line_count=18,
        )
        page.blocks = [block]

        validation = _validate_output([page], [block], [], [])
        issue_codes = {issue.code for issue in validation["issues"]}

        self.assertFalse(validation["passed"])
        self.assertIn("undersegmented_page", issue_codes)
        self.assertIn("embedded_heading", issue_codes)
        self.assertEqual(_compute_overall_confidence([block], validation), 0.0)

    def test_validate_output_skips_undersegmented_flag_for_sparse_plain_page(self) -> None:
        lines = [
            LineNode(id=f"plain-{i}", page=1, text=f"Simple line {i}", token_ids=[f"t{i}"], x0=40.0, y0=740.0 - (i * 18), x1=220.0, y1=750.0 - (i * 18), font_size=10.5, is_bold=False, indent=40.0)
            for i in range(12)
        ]
        block = BlockNode(
            id="b-plain-0",
            page=1,
            text="\n".join(line.text for line in lines),
            line_ids=[line.id for line in lines],
            x0=40.0,
            y0=542.0,
            x1=220.0,
            y1=750.0,
            font_size=10.5,
            is_bold=False,
            column_id=0,
            line_count=len(lines),
        )
        page = PageIR(number=1, width=612.0, height=792.0, characters=[], character_graph={}, tokens=[], lines=lines, blocks=[block])

        validation = _validate_output([page], [block], [], [])
        issue_codes = {issue.code for issue in validation["issues"]}

        self.assertTrue(validation["passed"])
        self.assertNotIn("undersegmented_page", issue_codes)
        self.assertIn("oversized_paragraph", issue_codes)

    def test_wrapped_resume_bullet_merges_and_prefers_list_role(self) -> None:
        lines = [
            LineNode(id="bullet", page=1, text="● Designed extraction heuristics for better code", token_ids=["t1"], x0=44.0, y0=664.0, x1=420.0, y1=674.0, font_size=10.5, is_bold=False, indent=44.0),
            LineNode(id="continuation", page=1, text="maintainability.", token_ids=["t2"], x0=72.0, y0=652.0, x1=190.0, y1=662.0, font_size=10.5, is_bold=False, indent=72.0),
        ]

        _assign_columns(lines, 612.0, 792.0)
        blocks = _build_blocks(1, lines, TEST_TUNING)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(
            blocks[0].text,
            "● Designed extraction heuristics for better code\nmaintainability.",
        )

        page = PageIR(number=1, width=612.0, height=792.0, characters=[], character_graph={}, tokens=[], lines=lines, blocks=blocks)
        _annotate_block_features([page], blocks)
        votes, _ = _run_deterministic_strategies([page], blocks)
        _apply_consensus(blocks, votes)

        self.assertEqual(blocks[0].features["bullet_signal"], 1.0)
        self.assertEqual(blocks[0].role, "list")

    def test_structured_item_signal_is_document_neutral(self) -> None:
        self.assertTrue(_looks_like_structured_item_line("Invoice Period: Jan 2024 - Feb 2024"))
        self.assertTrue(_looks_like_structured_item_line("Name | Role | Location"))
        self.assertFalse(_looks_like_structured_item_line("This is a regular narrative paragraph about project delivery."))

    def test_build_blocks_splits_structured_fields_from_paragraph(self) -> None:
        lines = [
            LineNode(id="field-1", page=1, text="Invoice Period: Jan 2024 - Feb 2024", token_ids=["t1"], x0=44.0, y0=700.0, x1=280.0, y1=710.0, font_size=10.5, is_bold=False, indent=44.0),
            LineNode(id="field-2", page=1, text="Amount Due: USD 1240", token_ids=["t2"], x0=44.0, y0=686.0, x1=220.0, y1=696.0, font_size=10.5, is_bold=False, indent=44.0),
            LineNode(id="body", page=1, text="Payment is due within thirty days of receipt.", token_ids=["t3"], x0=44.0, y0=664.0, x1=340.0, y1=674.0, font_size=10.5, is_bold=False, indent=44.0),
        ]

        _assign_columns(lines, 612.0, 792.0)
        blocks = _build_blocks(1, lines, TEST_TUNING)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, "Invoice Period: Jan 2024 - Feb 2024\nAmount Due: USD 1240")
        self.assertEqual(blocks[1].text, "Payment is due within thirty days of receipt.")


if __name__ == "__main__":
    unittest.main()