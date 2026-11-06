---
name: setup
description: Install and configure ContextRead (the `contextread` package) so it can index PDFs and answer questions from them. Covers the install, the .env (model provider, model names, spend cap), Postgres and storage through Docker, the database tables, and a first indexed document and answer. Use this whenever someone wants to install, set up, configure or start using ContextRead or `contextread`, connect it to their model provider or gateway, change its model names, or when `contextread check` fails, a command says a setting is missing, or the database or storage cannot be reached.
---

# Setting up ContextRead

ContextRead indexes each PDF into a tree of sections (titles, page ranges, summaries) and answers questions by
reading that tree and opening the pages that hold the answer. It needs four things: the Python package, an
OpenAI-compatible model endpoint, Postgres, and S3-compatible storage. The last two come from a
docker-compose file that `contextread init` writes.

Work through the steps in order and run each check before moving on. Most problems only show up as a
confusing error several steps later.

## 1. Prerequisites

- Python 3.12 or newer (`python3 --version`).
- Docker with compose (`docker compose version`).
- Access to the repository. It is private, so the person needs GitHub access: an SSH key or a logged-in
  `gh`/git credential helper.
- Optional: Tesseract (`apt install tesseract-ocr` / `brew install tesseract`). It is the fallback when the
  vision model fails on a scanned page. Without it, such a page is left without text.

## 2. Install

Prefer an isolated install so the package's dependencies (about 500 MB) don't mix with other projects:

```bash
uv tool install "git+ssh://git@github.com/janstylewis-wq/contextread.git"   # or a fixed version: ...contextread.git@v0.3.0
# without uv:
python3 -m venv ~/.venvs/contextread && ~/.venvs/contextread/bin/pip install "git+ssh://git@github.com/janstylewis-wq/contextread.git"
```

Confirm with `contextread --help`.

## 3. Model names

Before writing any settings, ask which provider they use and which models it offers, then pick one for each
job in the table under step 5 (what each model does, and what to pick). The defaults (`gpt-6-luna`, `gpt-4o-mini`) are the names on the gateway this was built
against, and most providers won't have them.

## 4. A folder with settings

Pass the chosen names to `init` as flags. When you run it there is no terminal to answer its questions, so
without flags it writes the defaults:

```bash
mkdir my-docs && cd my-docs
contextread init --chat-model <name> --summary-model <name> --vision-model <name> --tree-model <name> \
  --reasoning-models <prefixes>
# writes .env (random database/storage passwords, free ports) and docker-compose.yml
```

`contextread` reads `.env` from the folder it runs in, so run every later command from this folder.
Variables already set in the shell take precedence over the file. Each folder is its own Docker project, named
after the folder.

## 5. Provider key

`.env` holds secrets. Don't print it or paste it into the conversation, and don't ask the person to paste
their API key into the chat. Ask them to open `.env` in their editor and fill in:

- `OPENAI_API_KEY`: their key.
- `OPENAI_BASE_URL`: leave empty for api.openai.com. For anything else (Azure OpenAI, OpenRouter, a company
  gateway such as LiteLLM, a local server), set that endpoint's OpenAI-compatible URL.
- `LLM_BUDGET_USD`: a hard cap on model spend (default 10). Calls are refused once it is reached.

You may edit the non-secret lines yourself, such as model names and limits, after showing the change.

The model settings (`init`'s flags in brackets):

| Setting | Job | What to pick |
|---|---|---|
| `CHAT_MODEL` (`--chat-model`) | Answers questions: reads the tree, picks pages | The strongest model they can afford; a reasoning model did clearly best |
| `LLM_DEFAULT_MODEL` (`--summary-model`) | Section summaries and document descriptions | A small, cheap, fast model |
| `OCR_VLM_MODEL` (`--vision-model`) | Reads scanned or image-only pages | Must accept images; `off` = Tesseract only |
| `TREE_EXPAND_MODEL` (`--tree-model`) | Finds headings the layout missed, splits long sections | A capable model; `off` = layout-only tree |
| `REASONING_MODELS` (`--reasoning-models`) | Name prefixes of reasoning models, comma-separated (for example `o3,o4,gpt-5`) | These are sent `reasoning_effort` instead of `temperature`; a wrong value makes those calls fail |

Why the chat model matters most: on legal filings, a small model answered 47 of the 56 questions a reasoning
model answered on the same documents, and took more steps to do it.

## 6. Database and storage

```bash
docker compose up -d     # Postgres on PG_PORT (5434), storage on S3_PORT (9100)
contextread migrate      # creates the tables
```

If a port is taken (`address already in use`), change `PG_PORT` or `S3_PORT` in `.env` and run
`docker compose up -d` again. The connection settings below them follow automatically.

## 7. Check everything

```bash
contextread check
```

It prints one line per part (settings, database, storage, each model) and exits non-zero if something
needed is broken. Each model gets one tiny call, a fraction of a cent. Common failures:

- `settings: ValidationError ... (is .env here?)`: the command was run outside the folder with `.env`.
- `database: ... connection refused`: run `docker compose up -d`, or check `PG_PORT`.
- `database: schema ... run contextread migrate`: do that.
- A model `FAIL` with "not found" or "not allowed": the name doesn't exist on this provider or key.
- A model `FAIL` mentioning `temperature` or `reasoning`: add the model's name prefix to `REASONING_MODELS`,
  or remove it from there if it is not a reasoning model.

Re-run until every needed line is `ok`. A `note` line is information, not a failure.

## 8. First document and answer

```bash
contextread index some.pdf               # prints the document id, cost and the tree
contextread ask "a question about it" --doc <id>
contextread show <id> --summaries        # the tree with each section's summary
```

For a sense of cost: a 22-page earnings release took about $0.004 to index with a small summary model.
Scanned pages cost more, because each one is a vision call. Before indexing a large batch, index one
typical document and multiply its cost.

When this works, suggest measuring answers on their own documents with the `evaluate` skill.
