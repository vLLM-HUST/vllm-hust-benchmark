"""Preserve the pinned DocVQA validation records, not a replacement task."""

from pathlib import PurePosixPath


def verify_question(row):
    """Validate original field types without coercing answers or images."""
    for name in (
        "questionId",
        "question",
        "ucsf_document_id",
        "ucsf_document_page_no",
        "data_split",
    ):
        if not isinstance(row.get(name), str):
            raise ValueError("DocVQA requires original string fields")
    if type(row.get("docId")) is not int:
        raise ValueError("DocVQA requires original integer docId")
    for name in ("question_types", "answers"):
        values = row.get(name)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("DocVQA requires original string lists")
    image = row.get("image")
    if not isinstance(image, dict):
        raise ValueError("DocVQA requires original image struct")
    content, relative = image.get("bytes"), image.get("path")
    if not isinstance(content, bytes) or not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("DocVQA requires original PNG bytes")
    if (
        not isinstance(relative, str)
        or not relative
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError("DocVQA requires original relative image path")
