"""Versioned MMLU-Pro scoring correction for archived streaming responses."""

import re


_EXPLICIT_CHOICE = re.compile(
    r"\b(?i:(?:final[ \t]+)?answer(?:[ \t]+is\b|[ \t]*[:：]))"
    r"\s*(?:\*{1,2}|_{1,2}|\x60{1,3})?[ \t]*"
    r"(?:\([ \t]*([A-Ja-j])[ \t]*\)|([A-J])(?!\w))"
)
_LAST_STANDALONE_CHOICE = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)


def declared_answers(text: str) -> list[str]:
    """Return explicit A–J choices in the order the response declared them."""
    return [
        (match.group(1) or match.group(2)).upper()
        for match in _EXPLICIT_CHOICE.finditer(text)
    ]


def extract_answer(text: str) -> str | None:
    """Use the final explicit answer, retaining the frozen standalone fallback."""
    choices = declared_answers(text)
    if choices:
        return choices[-1]
    match = _LAST_STANDALONE_CHOICE.search(text)
    return match.group(0) if match else None

