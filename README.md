# ContextRead

Traceable answers from long documents. Each PDF is indexed into a tree of sections (titles, page ranges,
summaries). A chat agent reads that tree, opens the pages that hold the answer, and answers from what it
read, with optional citations to the exact blocks. The same index serves a REST API and an MCP server.

## Install

Needs Python 3.12+, Docker (for Postgres and object storage) and access to this repository.

```bash
pip install "git+https://github.com/janstylewis-wq/contextread.git"
# or a fixed version
pip install "git+https://github.com/janstylewis-wq/contextread.git@v0.2.0"
```

Scanned pages are read by a vision model. If that call fails, Tesseract is the fallback, so install it too
(`apt install tesseract-ocr`, `brew install tesseract`). Without it, such a page is left without text.

## First run

```bash
mkdir my-docs && cd my-docs
contextread init                 # writes .env (random passwords filled in) and docker-compose.yml
# put OPENAI_API_KEY (and OPENAI_BASE_URL for a gateway) in .env
docker compose up -d             # Postgres on 5434, S3-compatible storage on 9100 (PG_PORT, S3_PORT in .env)
contextread migrate              # create the tables
contextread index report.pdf     # index a PDF: prints its id, cost and tree
contextread ask "What was revenue in Q1?" --doc <id>
```

`contextread` reads `.env` from the folder you run it in; variables already set in the environment win.

## Commands

| Command | What it does |
|---|---|
| `contextread init [DIR]` | Write `.env` and `docker-compose.yml` (`--force` overwrites) |
| `contextread migrate` | Create or upgrade the database tables |
| `contextread index FILE.pdf` | Index a PDF and print its tree (`--show-summaries` for the first few summaries) |
| `contextread list` | Indexed documents with their ids and status |
| `contextread ask "QUESTION" --doc ID` | Answer from one or more documents (`--doc` repeats; `--cite` adds block citations) |
| `contextread serve` | API server: REST, chat completions, MCP (`--host`, `--port`, `--reload`) |
| `contextread worker` | Indexing worker for documents uploaded through the API |
| `contextread create-key --email E` | Create a user and an API key for the API and MCP server |

## Settings

All settings go in `.env`. The ones you are most likely to change:

| Setting | Default | What it controls |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | | Any OpenAI-compatible endpoint |
| `LLM_BUDGET_USD` | `10` | Hard cap on model spend; calls are refused once it is reached |
| `CHAT_MODEL` | `gpt-6-luna` | Model that answers questions (reasoning models run at low effort) |
| `LLM_DEFAULT_MODEL` | `gpt-4o-mini` | Section summaries and document descriptions |
| `OCR_VLM_MODEL` | `gpt-6-luna` | Reads scanned and image pages (`off` = Tesseract only) |
| `TREE_EXPAND_MODEL` | `gpt-6-luna` | Splits sections over 5 pages into their headings or 5-page parts (`off` = layout tree only) |
| `LLM_TIMEOUT_SECONDS` | `90` | A stalled model call is cut off and retried (twice) |
| `LLM_CACHE` | `true` | Identical model requests are answered from `~/.cache/contextread/llm` |
| `PAGE_LIMIT`, `CHAT_LIMIT` | `0` | Per-user limits for the API (0 = none) |

Default model names are those of the gateway this was built against. With api.openai.com directly, set
models your account has, and list your reasoning models' name prefixes in `REASONING_MODELS` (for example
`o3,o4,gpt-5`): they are sent `reasoning_effort` instead of `temperature`.

### Tuning

The defaults were measured on legal filings and credit agreements. Indexing settings apply to documents
indexed after the change; re-index a document to apply them to it.

| Setting | Default | What it controls |
|---|---|---|
| `SECTION_HEADINGS_PAGES` | `5` | A section over this many pages is checked for headings the layout missed |
| `HEADINGS_WINDOW_PAGES` | `10` | Pages read per call in that check |
| `SECTION_SPLIT_PAGES`, `SECTION_SPLIT_CHARS` | `10`, `20000` | A section still over this many pages and characters is split into parts |
| `SPLIT_PART_PAGES` | `5` | Pages per part; each part is titled and summarised by what it covers |
| `SUMMARY_INPUT_CHARS` | `8000` | How much of a section its summary is written from |
| `SCAN_IMAGE_SHARE` | `0.7` | A page one image covers this much of is read as a scan |
| `SCAN_INK_WITHOUT_TEXT` | `0.25` | A page this much of whose ink has no text behind it is read by the vision model |
| `CHAT_EFFORT` | `low` | Reasoning effort of a reasoning chat model; higher was slower for the same answers |
| `CHAT_MAX_TURNS` | `8` | Model turns per question; the last one has to answer |
| `CHAT_STRUCTURE_CHARS` | `400000` | The most of a tree the model sees at once; a larger tree keeps titles and pages, shortens summaries |
| `CHAT_PAGES_PER_READ`, `CHAT_READ_CHARS` | `10`, `80000` | Pages, and characters, one page read returns |
| `CHAT_ANSWER_MAX_TOKENS` | `1500` | Output limit of each chat turn |

## How it works

1. **Pages.** Each page's text layer is split into blocks (headings, paragraphs, lists, tables) with their
   positions. A page with no usable text (a scan, or text drawn as outlines) is read by the vision model in
   one call, which returns its blocks and their positions.
2. **Tree.** Headings become a tree of sections. A section over 5 pages is checked for headings the layout
   missed. One over 10 pages without headings is split into 5-page parts titled by what they cover.
   Every section gets a short summary.
3. **Answers.** The chat agent sees the whole tree, picks pages, reads them, and is told which pages it has
   not read yet. A reply that only says what it will read next is sent back to continue.

## Development

```bash
git clone https://github.com/janstylewis-wq/contextread.git && cd contextread
uv sync --extra dev
contextread init && docker compose up -d && contextread migrate
.venv/bin/pytest                 # uses a throwaway database (contextread_test); no billed model calls
.venv/bin/pytest -m live         # one small billed call through the budget guard
.venv/bin/pre-commit install     # ruff on every commit
```
