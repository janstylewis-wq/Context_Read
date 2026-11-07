# How ContextRead reads a document

## Contents
1. Pages: from PDF to text
2. Tree: from text to sections
3. Summaries
4. Answering: what the agent sees
5. Settings by symptom

## 1. Pages: from PDF to text

Each page's text layer is split into blocks (headings, paragraphs, lists, tables) with their positions. A page
goes to the vision model (`OCR_VLM_MODEL`) instead when its text layer can't be trusted:

- it has under 20 characters of text (no text layer);
- one image covers `SCAN_IMAGE_SHARE` (0.7) of the page, so it is a scan, even with an OCR layer on top;
- it has 50 or more small letter-sized vector shapes, or at least `SCAN_INK_WITHOUT_TEXT` (0.25) of its dark
  pixels have no text character under them (text drawn as outlines).

The vision model returns the page's blocks with positions in one call. If that call fails or is `off`,
Tesseract reads the page; if Tesseract isn't installed either, the page has no text. `contextread show`
reports each page's read mode: `text`, `vlm` or `image`.

The two thresholds decide cost as well as quality: each vision page is a model call with an image. Lowering
them sends more pages to the vision model.

## 2. Tree: from text to sections

`contextread show` says where a document's outline came from (`outline from:`):

| Value | Meaning |
|---|---|
| `contents` | Its table of contents |
| `detected` / `hybrid` | Layout headings |
| `bookmarks` | The PDF's own bookmarks |
| `headings` | Title blocks of pages the vision model read (mostly scanned documents) |
| `pages` / `pages+expand` | No structure found: page groups, then expand |

0. **Table of contents.** When `CONTENTS_OUTLINE` is on (the default), the first 20 pages are checked for a
   contents page, one `TREE_EXPAND_MODEL` call per page. If the contents give page numbers, each entry is
   placed on its real page: printed numbers are shifted by the offset between printed and real pages, each
   entry is checked to appear on its page (one call per entry), and misplaced entries are corrected. If more
   than 60% are found, those entries are the tree, and steps 1-3 are skipped. Long sections are still split
   in step 4. This is how PageIndex classic builds a tree. A contract's contents list every section,
   including ones whose headings are set like body text.
1. **Layout headings.** Headings found by layout analysis (font size, weight, numbering) become the tree.
   Headings set like body text, common in contracts and filings, are often missed.
2. **Expand.** `TREE_EXPAND_MODEL` reads sections the layout left whole and splits them into their
   subsections.
3. **Headings pass.** Every leaf section over `SECTION_HEADINGS_PAGES` (5) pages is read in
   `HEADINGS_WINDOW_PAGES` (10)-page calls for headings printed in it. A heading is kept only where it is
   printed on the page given. This recovers whole sections that the layout folded into the one before.
4. **Long-section split.** A leaf still over `SECTION_SPLIT_PAGES` (10) pages, with at least
   `SECTION_SPLIT_CHARS` (20,000) characters of text, is cut into `SPLIT_PART_PAGES` (5)-page parts, each
   starting on the page the one before ends on. Each part is titled and summarised by the entries that begin
   in it, for example "Definitions: Investment to Material Subsidiary". This is how a definitions article, a
   glossary, a schedule or an exhibit list becomes findable. The character minimum keeps runs of signature
   pages or blank forms from being split.

If none of this yields a structure, pages are grouped ten at a time.

## 3. Summaries

Every section gets a summary of up to about 60 words from `LLM_DEFAULT_MODEL`, written from the first
`SUMMARY_INPUT_CHARS` (8,000) characters of its own text. A parent's summary also sees its children's titles.
The document gets a one-paragraph description. Summaries are what the agent uses to decide where to look, so a
vague one hides its section.

## 4. Answering: what the agent sees

`CHAT_MODEL` runs a tool loop, at `CHAT_EFFORT` (low) effort if it is a reasoning model:

- `get_document_structure`: the whole tree (titles, page ranges, summaries). When it is larger than
  `CHAT_STRUCTURE_CHARS` (400,000 characters, about 100k tokens), summaries are shortened, deepest first: to
  160 characters, then 60, then none. Titles and page ranges are always kept.
- `get_page_content`: up to `CHAT_PAGES_PER_READ` (10) pages per call, as blocks. When the text is over
  `CHAT_READ_CHARS` (80,000), whole pages are dropped from the end with a note naming them.
- After each read the agent is told which pages it hasn't read yet.
- A reply that only announces more reading is sent back once or twice with "continue".
- After `CHAT_MAX_TURNS` (8) turns the last one has no tools, so the agent must answer from what it has.
- Each turn's output is capped at `CHAT_ANSWER_MAX_TOKENS` (1500).

## 5. Settings by symptom

Indexing settings apply to documents indexed after the change: run `contextread reindex <doc>` for each one
affected. Chat settings apply on the next question.

| Symptom | Setting to try | Direction | Cost of the change |
|---|---|---|---|
| Answer inside a long, unsplit section | `SECTION_SPLIT_PAGES`, `SECTION_SPLIT_CHARS` | lower | re-index; more parts, more summary calls |
| Parts too coarse to point at the page | `SPLIT_PART_PAGES` | lower (3-4) | re-index; more parts |
| Printed headings missing from the tree | `SECTION_HEADINGS_PAGES`, `TREE_EXPAND_MODEL` | lower; make sure it isn't `off` | re-index; more expand calls |
| Sections on the wrong pages, from a table of contents (`outline from: contents`) | `CONTENTS_OUTLINE` | `false` for that kind of document, then compare | re-index; the layout tree instead |
| A document has a table of contents but `outline from:` is not `contents` | none | it may lack page numbers, start after page 20, or have most entries not found on their page | report it with the document |
| Vague or placeholder summaries | `LLM_DEFAULT_MODEL` | stronger model | re-index; summary cost rises |
| Fact deep in a long section missing from its summary | `SUMMARY_INPUT_CHARS` | higher | re-index; longer summary calls |
| Garbled text on scan-like pages | `SCAN_IMAGE_SHARE`, `SCAN_INK_WITHOUT_TEXT` | lower | re-index; more vision calls |
| Huge tree with summaries cut to nothing | `CHAT_STRUCTURE_CHARS` | higher, within the chat model's context | more input tokens per question |
| Agent stops early or explores too little | `CHAT_EFFORT`, `CHAT_MAX_TURNS`, `CHAT_MODEL` | medium; 10-12; stronger | slower and costlier per question |
| Reads cut short ("pages were not shown") | `CHAT_READ_CHARS` | higher | more input tokens per read |

On the sets this was built on, `CHAT_EFFORT` medium or high was slower and gave the same answers. Treat a
higher effort as something to measure, not a sure gain.
