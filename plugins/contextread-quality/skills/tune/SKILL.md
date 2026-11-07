---
name: tune
description: Improve ContextRead's answers on the user's documents by changing its settings one at a time and measuring each change against a baseline with `contextread eval`. Changes that help are kept and the rest are reverted. Use this whenever someone wants better accuracy, fewer misses, faster or cheaper answers, wants to adapt ContextRead to their kind of documents (contracts, reports, scans, manuals), asks which settings to change, or wants to try a different model, even if they only say "make it work better on my files".
---

# Tuning ContextRead

Tuning means changing one setting, measuring, and keeping the change only if the score moved by more than
noise. Several changes at once can't be told apart, and a change judged on a feeling or on one question often
helps one document while quietly breaking another.

## Before changing anything

You need three things, all from the same folder as `.env`:

1. **A questions file** of 30 to 60 reviewed questions from the documents that matter. Without one, use the
   `evaluate` skill first.
2. **A baseline results file** from `contextread eval run questions.json` on the current settings.
3. **A reason for the change.** Use the `diagnose` skill on the baseline's misses. Tune toward a cause shared
   by several misses, not one.

Also agree on what "better" means. Usually it is more questions right, but it can be the same score faster or
cheaper. Agree on how much may be spent: each run costs about $0.002 per question, and re-indexing costs about
what indexing did.

## The loop

1. **Pick one change** with its reason, for example: "`SECTION_SPLIT_PAGES` 10 → 6: four misses sit inside
   7-9-page sections that were never split." The tables in `../diagnose/references/how-it-works.md` map
   symptoms to settings and say what each change costs. Settings not set in `.env` use their defaults, which
   are listed in the README's Tuning section.

2. **Apply it in `.env`.** Show the line before and after. Edit only that line, and never print the secrets
   in the file.

3. **Re-index if it is an indexing setting.** These are `SECTION_*`, `HEADINGS_WINDOW_PAGES`,
   `SPLIT_PART_PAGES`, `SUMMARY_INPUT_CHARS`, `SCAN_*`, `LLM_DEFAULT_MODEL`, `OCR_VLM_MODEL` and
   `TREE_EXPAND_MODEL`. For each document in the questions file:
   ```bash
   contextread reindex <doc id or name>
   ```
   It rebuilds the document under its same id, so the questions file still points at it. Requests that didn't
   change are answered from the cache at no cost. Chat settings (`CHAT_*`) need no re-index.

4. **Measure:**
   ```bash
   contextread eval run questions.json --against <baseline results file>
   ```
   The report ends with the questions that were fixed and the ones that broke.

5. **Decide:**
   - Runs of the same 60 questions on the same settings differ by about 1 to 2 answers. A gain of 1 or 2 is
     noise. Before keeping a small gain, run both settings once more and compare totals across the runs.
   - Read the broke list as well as the fixed list. Four fixed and three broken is not a real improvement.
     Check the broken ones with `diagnose`, because they show what the change cost.
   - Keep the change if it clearly helped and broke nothing important. Otherwise restore the old line (and
     re-index again if it was an indexing setting).
   - The new results file becomes the baseline for the next change.

6. **Write it down** in `TUNING.md` in the folder: the date, the setting with its old and new values, the
   reason, before → after (right, answer page read, time, cost), and kept or reverted. When someone later asks
   why a setting has an odd value, the answer is there.

## Avoiding a fit to the questions

After several kept changes, the settings can fit these particular questions rather than the documents.
Before calling tuning done, write a fresh set (`contextread eval make --seed 1 --out questions-check.json`),
review it, and run it on the old and the new settings. The gain should hold up on questions the tuning never
saw. If it doesn't, keep only the changes that still help.

## Changes that are not settings

When a cause can't be helped by a setting, say so rather than pushing the numbers around. Examples are a
planning note returned as the answer, or a document type the tree builder handles badly. Describe it with an
example question so it can be reported. A stronger model for one job (`CHAT_MODEL`, `LLM_DEFAULT_MODEL`,
`OCR_VLM_MODEL`) counts as a setting and is measured the same way. Check that the new name answers with
`contextread check` first.
