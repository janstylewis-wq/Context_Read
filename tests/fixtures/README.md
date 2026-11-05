# Test fixtures

- `q1-fy25-earnings.pdf`: The Walt Disney Company Q1 FY2025 earnings release, a public document,
  taken from the MIT-licensed PageIndex repository's `examples/documents/`.
- `pageindex_reference_blocks.json`: the blocks PageIndex's hosted product returned for pages 1
  and 3 of that PDF, recorded on 2026-11-01. Used as a reference for block
  granularity and content, not for exact ids or coordinates.
- `court-order-excerpt.pdf`: pages 4, 5 and 13 of a public U.S. District Court (D.S.C.) order,
  case 4:20-cv-00067-SAL, ECF No. 156. Much of its visible text, including most bold headings, is
  drawn as vector outlines with no text-layer characters behind it.
- `court_order_excerpt_vlm.json`: gpt-6-luna's Markdown for those three pages, recorded on
  2026-11-02 so the tests can replay the vision model without calling it.
