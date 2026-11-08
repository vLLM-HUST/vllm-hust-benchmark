"""Qualify software/tokenizer only; never execute SQL, original tasks or models."""

from importlib import metadata
import json
import platform
import sqlite3

PINS = {
    "sqlparse": "0.5.3",
    "nltk": "3.9.2",
    "tqdm": "4.67.1",
    "click": "8.5.0",
    "cloudpickle": "3.1.2",
    "joblib": "1.6.0",
    "regex": "2026.9.29",
}


def verify():
    import nltk
    import sqlparse

    versions = {name: metadata.version(name) for name in PINS}
    if versions != PINS:
        raise ValueError("disclosed Spider integration package pins differ")
    # Tokenization is a software dependency control, not a SQL execution test.
    if nltk.word_tokenize("software tokenizer qualification") != [
        "software",
        "tokenizer",
        "qualification",
    ]:
        raise ValueError("required English tokenizer resource unavailable")
    if not sqlparse.parse("SELECT 1"):
        raise ValueError("sqlparse dependency unavailable")
    return {
        "status": "SOFTWARE_ONLY_QUALIFIED_NATIVE_SCORER_PENDING",
        "integration_pins_not_author_lock": versions,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "sqlite_library": sqlite3.sqlite_version,
        "original_tasks_executed": 0,
        "sql_statements_executed": 0,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
