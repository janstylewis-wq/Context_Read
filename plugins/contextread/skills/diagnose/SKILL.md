---
name: diagnose
description: Find out why ContextRead gave a wrong or missing answer on a document. Trace the question through the document's tree, the pages the agent opened, and the page text, until the cause is named (a heading the layout missed, a long section never split, a vague summary, a scan read badly, a read cut short, an agent that stopped early, or a bad question or judge). Use this whenever someone says ContextRead answered wrongly, couldn't find something that is in the document, gave a strange answer, or asks why an eval question failed; and after `contextread eval run` reports misses. Even a single bad answer is worth diagnosing this way rather than guessing.
---

# Why ContextRead missed

Each wrong answer has one cause somewhere along a short path: the page's text, then the tree built from the
pages, then the agent reading that tree. Walk the path with evidence instead of guessing. The fix for a tree
problem is very different from the fix for an agent problem, and a wrong guess leads to tuning the wrong
setting.

Read `references/how-it-works.md` when you need the details of a step. It explains how pages are read, how
the tree is built, and what the agent is shown.

This skill names the cause and proposes the change; it changes nothing itself. Settings are changed and
measured with the `tune` skill.

## What you need

- The question, the expected answer, and the page holding it. From an eval results file, each entry has
  `doc_name`, `page`, `question`, `answer`, `answer_given`, `pages_read`, `steps` and `outcome`.
- For a question someone just asked, re-ask it: `contextread ask "..." --doc <id>`. The footer lists the
  model, the time, the cost and the pages read. Ask the person which page holds the answer, or find it.

## Steps

1. **Is the question fair, and is the judge right?** Compare the given answer with the expected one. Then
   check whether the answer is also stated on a page the agent did read: if it is, "missed page" only means
   the agent found it elsewhere. Look for an ambiguous question or a wrong expected answer
   (`contextread show <doc> --pages <page>`). Whichever applies, fix the question, not ContextRead.

2. **Is the page's text right?** Run `contextread show <doc> --pages <page>`. The header says how the page was
   read: `text` (the PDF's own text layer), `vlm` (read by the vision model) or `image` (Tesseract fallback,
   or no text).
   - Missing, garbled or out-of-order text while read as `text`: the PDF's text layer is bad (common with
     scans that carry someone else's OCR, or text drawn as outlines). Scan detection let it through.
   - A table flattened into an unreadable run of numbers: the same problem, on a table.
   - Read as `image` with poor text: the vision model failed or is `off`, and Tesseract read it.
   - Text fine: go on.

3. **Does the tree lead to the page?** Run `contextread show <doc> --summaries` and find the section whose page
   range holds the answer page. Then ask whether someone reading only titles and summaries would open it for
   this question.
   - The page sits inside a long section (many pages) whose title and summary describe something else, or
     only its first pages: the section was never split. Check its length against `SECTION_SPLIT_PAGES`, and
     its text against `SECTION_SPLIT_CHARS`.
   - A heading printed on the page (look at the page text) is not in the tree: the layout missed it, and the
     headings pass didn't recover it. Is `TREE_EXPAND_MODEL` off, or the section under
     `SECTION_HEADINGS_PAGES`? Check `outline from:` in the `show` header. A document whose tree comes
     from its `contents` has exactly the headings its table of contents lists.
   - The section is right but its summary is vague, generic, or full of placeholders such as "$X million": a
     weak summary. The summary model is `LLM_DEFAULT_MODEL`, and it reads the section's first
     `SUMMARY_INPUT_CHARS` characters, so a fact deep in a long section may never reach it.
   - A very large document (hundreds of sections): the agent sees summaries shortened to fit
     `CHAT_STRUCTURE_CHARS`. Check whether that section's summary was cut to nothing.
   - Tree fine: go on.

4. **Did the agent read well?** Look at `pages_read` and `steps`.
   - It opened the right section's pages but not the answer page: the section is too coarse. Treat it as a
     tree problem (step 3).
   - Few steps, and it never looked where the tree points: it gave up early. Look at `CHAT_EFFORT` and
     `CHAT_MAX_TURNS`, or a weak `CHAT_MODEL`.
   - It read the page, but the reply says pages "were not shown": the read exceeded `CHAT_READ_CHARS` and was
     cut at whole pages.
   - The answer is a note about what to read next ("Need to check page 133") instead of an answer: the agent
     loop let a planning note through as the final answer. That's a ContextRead defect, not a setting; report
     it with the question.

## Report

For each miss, give one line with the cause and the evidence. For example: "p244 sits in 'Exhibits'
(p230-260), one 31-page section with a summary naming only its first exhibits; the layout found no headings
inside it, so it was never split." Then group the misses by cause, because a cause shared by several misses
is worth tuning. One that hits a single question may just be that question.

For each group, propose the smallest general change that would address it: a setting from the tables in
`references/how-it-works.md`, a better model for one job, or fixing the question. Also say what it would cost,
for example a re-index of the documents it affects. Then hand over to the `tune` skill to try it and measure
it. A change that fixes one document can break others, which is why nothing is changed before it is
measured.
