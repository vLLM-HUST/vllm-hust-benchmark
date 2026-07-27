from vllm_hust_benchmark.hf_publisher import _create_commit_on_branch


def test_create_commit_on_branch_uses_revision_when_supported() -> None:
    class RevisionApi:
        def __init__(self) -> None:
            self.revision = None

        def create_commit(
            self,
            repo_id,
            repo_type,
            operations,
            commit_message,
            revision=None,
        ):
            self.revision = revision
            return {"repo_id": repo_id, "repo_type": repo_type}

    api = RevisionApi()

    _create_commit_on_branch(
        api,
        repo_id="owner/repo",
        repo_type="dataset",
        branch="main",
        operations=[],
        commit_message="msg",
    )

    assert api.revision == "main"


def test_create_commit_on_branch_falls_back_to_branch_for_older_hf_api() -> None:
    class BranchApi:
        def __init__(self) -> None:
            self.calls = []

        def create_commit(self, **kwargs):
            self.calls.append(kwargs)
            if "revision" in kwargs:
                raise TypeError(
                    "create_commit() got an unexpected keyword argument 'revision'"
                )
            return kwargs

    api = BranchApi()

    commit_info = _create_commit_on_branch(
        api,
        repo_id="owner/repo",
        repo_type="dataset",
        branch="main",
        operations=[],
        commit_message="msg",
    )

    assert len(api.calls) == 2
    assert api.calls[0]["revision"] == "main"
    assert api.calls[1]["branch"] == "main"
    assert commit_info["branch"] == "main"
