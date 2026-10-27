"""Original QuALITY article/question relations, not prompt or workload synthesis."""

from copy import deepcopy


def question_rows(parents):
    """Expand release-order questions with explicit original parent/child positions."""
    seen = set()
    reserved = {"source_parent_row_index", "source_question_index"}
    for parent_index, parent in enumerate(parents):
        questions = parent.get("questions")
        if (
            not isinstance(questions, list)
            or not questions
            or not isinstance(parent.get("article"), str)
            or not parent["article"].strip()
            or not isinstance(parent.get("article_id"), str)
            or not parent["article_id"]
            or not isinstance(parent.get("set_unique_id"), str)
            or not parent["set_unique_id"]
            or reserved & parent.keys()
        ):
            raise ValueError("invalid original QuALITY article/question-set contract")
        context = {k: v for k, v in parent.items() if k != "questions"}
        for question_index, child in enumerate(questions):
            if (
                not isinstance(child, dict)
                or set(context) & child.keys()
                or reserved & child.keys()
            ):
                raise ValueError(
                    "QuALITY parent/question fields collide or are invalid"
                )
            question_id = child.get("question_unique_id")
            if (
                not isinstance(question_id, str)
                or not question_id
                or question_id in seen
            ):
                raise ValueError("QuALITY requires distinct original question IDs")
            seen.add(question_id)
            yield {
                **deepcopy(context),
                **deepcopy(child),
                "source_parent_row_index": parent_index,
                "source_question_index": question_index,
            }
