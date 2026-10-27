"""Original LoCoMo QA relations; no generated memories, prompts or timestamps."""

from copy import deepcopy
import re


def question_rows(parents):
    """Stream original questions with their entire original conversation."""
    seen = set()
    reserved = {"source_parent_row_index", "source_question_index"}
    for parent_index, parent in enumerate(parents):
        sample_id = parent.get("sample_id")
        conversation = parent.get("conversation")
        questions = parent.get("qa")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen
            or not isinstance(conversation, dict)
            or not isinstance(questions, list)
            or not questions
            or reserved & parent.keys()
        ):
            raise ValueError("invalid original LoCoMo conversation/QA contract")
        seen.add(sample_id)
        sessions = [key for key in conversation if re.fullmatch(r"session_\d+", key)]
        if not sessions:
            raise ValueError("LoCoMo conversation requires original sessions")
        for session in sessions:
            turns = conversation[session]
            if (
                not isinstance(turns, list)
                or not turns
                or not isinstance(conversation.get(session + "_date_time"), str)
            ):
                raise ValueError("invalid original LoCoMo session/date")
            for turn in turns:
                if not isinstance(turn, dict) or any(
                    not isinstance(turn.get(key), str) or not turn[key]
                    for key in ("speaker", "text", "dia_id")
                ):
                    raise ValueError("invalid original LoCoMo turn")
        for question_index, child in enumerate(questions):
            if (
                not isinstance(child, dict)
                or {"conversation", "sample_id", *reserved} & child.keys()
                or not isinstance(child.get("question"), str)
                or not child["question"].strip()
                or type(child.get("category")) is not int
                or child["category"] not in {1, 2, 3, 4, 5}
                or not ({"answer", "adversarial_answer"} & child.keys())
            ):
                raise ValueError("invalid original LoCoMo question relation")
            # Only the original conversation is input context. Parent QA siblings,
            # observations, session summaries and event gold do not enter a row.
            yield {
                "sample_id": sample_id,
                "conversation": deepcopy(conversation),
                **deepcopy(child),
                "source_parent_row_index": parent_index,
                "source_question_index": question_index,
            }
