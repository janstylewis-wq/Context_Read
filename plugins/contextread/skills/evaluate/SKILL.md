---
name: evaluate
description: Measure how well ContextRead answers questions on the user's own indexed documents with `contextread eval`. It writes or collects questions whose answer page is known, reviews them, runs them fresh, and explains the score and every miss. Use this whenever someone asks how accurate or reliable ContextRead is on their documents, wants a score or benchmark, wants to test it before relying on it, has a list of questions with known answers, or wants to compare before and after a change, even if they don't say "eval".
---

# Measuring ContextRead on your documents

The score is only as good as the questions behind it, so most of the care goes into the questions. Running
them is the easy part.

## 1. Ready?

Run `contextread check --no-models` from the folder with `.env`. If it fails, use the `setup` skill first.
Then run `contextread list` to see the indexed documents, and ask which ones to measure on if that isn't
clear. Mix the kinds of documents they really use.

## 2. Questions

There are two sources, and they can be combined.

**Questions the person already has.** Real questions with known answers measure best. Put them in a JSON file:

```json
[{"doc_name": "lease-2023.pdf", "question": "When does the lease end?", "answer": "31 March 2028", "page": 4}]
```

`doc_id` can replace `doc_name`. `page` is optional, but with it the report can tell whether a wrong answer
came from never opening the right page.

**Generated questions:**

```bash
contextread eval make --doc <id or name> --doc <...> --per-doc 5 --out questions.json   # no --doc: every indexed document
```

The command picks random pages with real text (the same `--seed` picks the same pages). For each page, a model
writes one question answered on that page, plus the short answer. Cost is a fraction of a cent per question.

Aim for 30 to 60 questions in total. Runs vary by about one to two questions in 60, so with 10 questions a
single answer moves the score by 10% and no comparison means much.

## 3. Review the questions

Generated questions go wrong in predictable ways, and each bad one makes the score noise. Read the file, and
for any question that looks doubtful run `contextread show <doc> --pages <page>` to see the page. Drop or fix:

- questions answerable from the document's title or general knowledge;
- vague ones that several pages could answer differently ("What is the total?");
- ones that ask two things at once;
- expected answers the page doesn't support, or that are cut short.

Tell the person what you dropped and why. If they know the documents, ask them to skim the file too.

## 4. Run

Before running, give the cost estimate and get an OK: about $0.002 per question with a reasoning chat model,
and less with a small one.

```bash
contextread eval run questions.json                       # writes results-<time>.json and prints the report
contextread eval run questions.json --against results-<earlier>.json   # also lists questions that changed
```

The cache is off by default, so answers are fresh, as a user would get them. Four questions run at once, at
about 6 to 12 seconds each.

## 5. Explain the result

The report starts with three numbers:
- **right**: answers that state the expected fact, as judged by a model;
- **answer page read**: how often the agent opened the page that holds the answer;
- **steps and time**: pages opened per question, and median seconds.

Each answer that isn't right is one of these:
- **missed page**: the answer page was never opened, so the tree led the agent elsewhere. Look at the tree
  first.
- **misread**: the page was opened and the answer is still wrong. The page's text, the judge or the expected
  answer is usually behind this.
- **wrong**: there is no page to tell which (hand-written question without one).
- **error**: the call failed; the message says why.

Before calling something a miss, check the judge: read "expected" against "given". An answer that states the
fact in other words, or answers a question that turned out to be ambiguous, isn't a real miss. Say so, and
fix or drop the question for next time.

Report in plain terms: the score, what the misses have in common, and which ones look like question or judge
problems rather than ContextRead problems. Keep the questions file and the results file name, because both
are needed to measure any change later. For the cause of each real miss, use the `diagnose` skill.
