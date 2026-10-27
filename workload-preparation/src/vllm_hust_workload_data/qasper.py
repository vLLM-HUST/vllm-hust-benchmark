"""Publisher v0.3 paper/question relations, not LongBench-derived prompts."""

from copy import deepcopy
import json


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate original JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("non-finite original JSON value")


def question_rows(path):
    """Keep full paper inputs and all original annotations on the reference side."""
    try:
        papers = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError):
        raise ValueError("invalid original QASPER JSON; payload suppressed") from None
    if not isinstance(papers, dict) or not papers:
        raise ValueError("original QASPER requires a paper-ID keyed object")
    seen = set()
    input_keys = {"title", "abstract", "full_text", "figures_and_tables"}
    for paper_index, (paper_id, paper) in enumerate(papers.items()):
        if (
            not isinstance(paper_id, str)
            or not paper_id
            or not isinstance(paper, dict)
            or not input_keys.issubset(paper)
            or not isinstance(paper.get("qas"), list)
            or not paper["qas"]
            or not isinstance(paper["title"], str)
            or not isinstance(paper["abstract"], str)
            or not isinstance(paper["full_text"], list)
            or not isinstance(paper["figures_and_tables"], list)
        ):
            raise ValueError("invalid original QASPER paper contract")
        for section in paper["full_text"]:
            if (
                not isinstance(section, dict)
                or "section_name" not in section
                or not isinstance(section.get("paragraphs"), list)
                or any(not isinstance(text, str) for text in section["paragraphs"])
            ):
                raise ValueError("invalid original QASPER section contract")
        for question_index, question in enumerate(paper["qas"]):
            if (
                not isinstance(question, dict)
                or not isinstance(question.get("question_id"), str)
                or not question["question_id"]
                or question["question_id"] in seen
                or not isinstance(question.get("question"), str)
                or not question["question"].strip()
                or not isinstance(question.get("answers"), list)
                or not question["answers"]
                or any(
                    not isinstance(ref, dict) or not isinstance(ref.get("answer"), dict)
                    for ref in question["answers"]
                )
            ):
                raise ValueError("invalid original QASPER question/reference relation")
            seen.add(question["question_id"])
            yield {
                "paper_id": paper_id,
                **{key: deepcopy(paper[key]) for key in input_keys},
                "question": question["question"],
                "question_id": question["question_id"],
                "answers": deepcopy(question["answers"]),
                "question_metadata": {
                    key: deepcopy(value)
                    for key, value in question.items()
                    if key not in {"question", "question_id", "answers"}
                },
                "paper_metadata": {
                    key: deepcopy(value)
                    for key, value in paper.items()
                    if key not in input_keys | {"qas"}
                },
                "source_paper_row_index": paper_index,
                "source_question_index": question_index,
            }
