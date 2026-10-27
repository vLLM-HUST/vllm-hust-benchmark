"""Publisher-manifest qualification for original Copilot session metadata.

This is asset integrity checking, not trace analysis or request synthesis.
"""

from datetime import date


def qualify_day_manifest(manifest: dict, members: list[dict]) -> dict:
    """Require every original shard, exact byte counts and original session counts.

    ``members`` contains independently measured path/bytes/rows metadata, never
    generated sessions. Archive compression bytes differ from inner shard bytes.
    """
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError("publisher day manifest must declare complete")
    day = manifest.get("date")
    try:
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("invalid publisher date") from None
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("publisher shard list is required")
    for field in ("n_sessions", "n_shards", "total_bytes"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise ValueError("invalid publisher totals")
    expected = {}
    for shard in shards:
        if not isinstance(shard, dict) or any(
            type(shard.get(field)) is not int or shard[field] < 0
            for field in ("shard_idx", "size_bytes", "n_sessions")
        ):
            raise ValueError("invalid publisher shard metadata")
        path = f"date={day}/shard-{shard['shard_idx']:04d}.jsonl.gz"
        if path in expected:
            raise ValueError("duplicate publisher shard")
        expected[path] = (shard["size_bytes"], shard["n_sessions"])
    if (
        len(expected) != manifest["n_shards"]
        or sum(v[0] for v in expected.values()) != manifest["total_bytes"]
        or sum(v[1] for v in expected.values()) != manifest["n_sessions"]
    ):
        raise ValueError("publisher manifest totals disagree")
    observed = {}
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise ValueError("invalid measured member")
        path = member["path"]
        if path in observed or any(
            type(member.get(field)) is not int or member[field] < 0
            for field in ("bytes", "rows")
        ):
            raise ValueError("duplicate or invalid measured member")
        observed[path] = (member["bytes"], member["rows"])
    if observed != expected:
        raise ValueError("original shards do not match publisher manifest")
    return {
        "date": day,
        "shards": len(expected),
        "sessions": manifest["n_sessions"],
        "compressed_shard_bytes": manifest["total_bytes"],
    }
