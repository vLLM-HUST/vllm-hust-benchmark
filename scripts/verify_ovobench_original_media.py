"""Check registered complete original clips; optional CPU video decode, no model.

Registered media is a partial input preparation scope, not benchmark completion.
No install, download, frame sampling, media rewriting or neural inference.
"""

import argparse
import json
from pathlib import Path

from prepare_ovobench_native_selection import prepare
from vllm_hust_workload_data.assets import asset_path, verify_file

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/ovobench-author-source.json"
)


def verify_archives(root, inventory):
    """Read-only publisher LFS verification of complete precut archive parts."""
    parts = [
        item
        for item in inventory["parts"]
        if item["path"].startswith("chunked_videos.tar.part")
    ]
    if not parts:
        raise ValueError("pinned original precut archive inventory required")
    return [
        verify_file(
            root,
            {
                "path": item["path"],
                "bytes": item["bytes"],
                "sha256": item["publisher_lfs_sha256"],
            },
        )
        for item in parts
    ]


def verify(members, selection, media_root, *, decode=False):
    expected = {item["path"]: item for item in selection["required_native_clips"]}
    if len(expected) != selection["required_native_clip_count"]:
        raise ValueError("complete distinct fixed-selection clip relation required")
    if len({item["path"] for item in members}) != len(members) or not members:
        raise ValueError("distinct registered original media identities required")
    if decode:
        import av
    results = []
    for item in members:
        original = expected.get(item["path"])
        if original is None or any(
            item[key] != original[key] for key in ("case_id", "task", "stage_index")
        ):
            raise ValueError("registered clip differs from fixed original selection")
        identity = verify_file(media_root, item)
        if decode:
            path = asset_path(media_root, item["path"])
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                dimensions = (
                    stream.codec_context.name,
                    stream.width,
                    stream.height,
                    str(stream.average_rate),
                    len(container.streams.audio),
                )
                if dimensions != (
                    item["video_codec"],
                    item["width"],
                    item["height"],
                    item["average_rate"],
                    item["audio_streams"],
                ):
                    raise ValueError("original media stream dimensions differ")
                count = sum(1 for _ in container.decode(video=0))
                if count != item["video_frames"] or (
                    stream.frames and count != stream.frames
                ):
                    raise ValueError("complete original video frame count differs")
            verify_file(media_root, item)
        results.append(
            {
                **identity,
                "case_id": item["case_id"],
                "full_video_decode_verified": decode,
            }
        )
    return {
        "source_id": "OVOBENCH",
        "status": "REGISTERED_ORIGINAL_MEDIA_VERIFIED_NOT_NATIVE_MODEL",
        "files": results,
        "registered_clip_count": len(members),
        "required_fixed_clip_count": selection["required_native_clip_count"],
        "all_required_clips_registered": len(members) == len(expected),
        "model_frame_sampler_qualified": False,
        "media_modified": False,
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="Also hash complete original precut parts against publisher LFS identities; large read-only I/O",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Use an existing PyAV decoder; no automatic installation",
    )
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(PROFILE.read_text("utf-8"))["media_archive_inventory"]
    members = inventory["qualified_original_members"]
    selection = prepare(args.data_root)
    if args.require_all and len(members) != selection["required_native_clip_count"]:
        parser.error(
            "fixed original media not all registered; no substitutions allowed"
        )
    report = verify(members, selection, args.media_root, decode=args.decode)
    if args.archive_root is not None:
        report["verified_publisher_archive_parts"] = verify_archives(
            args.archive_root, inventory
        )
        report["archive_member_to_every_extracted_clip_comparison_performed"] = False
    print(json.dumps(report))
