"""Section summaries are written from the section's whole text, subsections included, and the prompt asks for
facts from that text only."""

from fakes import completion

from contextread.indexing import tree


class Recorder:
    def __init__(self):
        self.prompts = []

    def chat(self, model, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        return completion("summary")


def _doc():
    return [
        {
            "title": "OTHER FINANCIAL INFORMATION",
            "start_index": 9,
            "end_index": 12,
            "text": "## OTHER FINANCIAL INFORMATION",
            "nodes": [
                {
                    "title": "Interest",
                    "start_index": 10,
                    "end_index": 10,
                    "text": "Net interest expense was $367 million.",
                },
                {
                    "title": "Taxes",
                    "start_index": 10,
                    "end_index": 11,
                    "text": "The effective tax rate was 27.8%.",
                },
            ],
        }
    ]


def test_a_parent_is_summarised_from_its_subsections_text():
    doc = _doc()
    assert tree.section_text(doc[0], 8000) == (
        "## OTHER FINANCIAL INFORMATION\n\nNet interest expense was $367 million.\n\nThe effective tax rate was 27.8%."
    )
    llm = Recorder()
    tree.summarize_nodes(doc, llm, model="m", workers=1)
    parent = next(p for p in llm.prompts if "Section title: OTHER FINANCIAL INFORMATION" in p)
    assert "$367 million" in parent and "27.8%" in parent and "Subsections: Interest; Taxes" in parent
    assert "never write placeholders" in parent


def test_section_text_stops_at_the_limit():
    assert len(tree.section_text(_doc()[0], 40)) == 40
