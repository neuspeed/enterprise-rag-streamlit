"""Tests for the sandbox dashboard helpers.

The UI itself is exercised by hand in a browser; what is worth asserting here is
the logic around it — payload assembly, upload sanitising and the branch label —
because those silently produce a malformed message to the pipeline if wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

# streamlit lives in the `ui` extra, which CI does not install. Skipping keeps the
# suite runnable on requirements.txt alone rather than forcing a 50 MB dependency
# onto every test job.
pytest.importorskip("streamlit")

from know_everything_ai.ui.app import (  # noqa: E402
    ACCEPTED_SUFFIXES,
    KB_MODE_APPEND,
    KB_MODE_NEW,
    SessionLoop,
    _requested_branch,
    append_target_problem,
    build_payload,
    build_sandbox_settings,
    save_uploads,
    upload_dir,
)

WEBHOOK = "http://webhook_sink:80/hook"


def test_sandbox_settings_allow_local_files() -> None:
    """Uploads are paths, and the production default refuses paths."""
    assert build_sandbox_settings().ALLOW_LOCAL_SOURCES is True


def test_upload_dir_is_under_the_working_tree(tmp_path: Path, monkeypatch) -> None:
    """Never DATA_DIR: in the stand it is the container path /app/data."""
    monkeypatch.chdir(tmp_path)

    target = upload_dir()

    assert target == tmp_path / "data" / "ui_uploads"
    assert target.is_dir()


def test_save_uploads_strips_directory_components(tmp_path: Path, monkeypatch) -> None:
    """A browser-supplied name must not be able to escape the upload directory."""
    monkeypatch.chdir(tmp_path)
    evil = SimpleNamespace(name="../../etc/passwd", getbuffer=lambda: b"data")

    saved = save_uploads([evil])

    assert len(saved) == 1
    written = Path(saved[0])
    assert written.parent == upload_dir()
    assert written.read_bytes() == b"data"
    assert ".." not in written.name


def test_save_uploads_accepts_multiple_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    files = [
        SimpleNamespace(name="a.pdf", getbuffer=lambda: b"1"),
        SimpleNamespace(name="b.txt", getbuffer=lambda: b"2"),
    ]

    saved = save_uploads(files)

    assert [Path(p).suffix for p in sorted(saved)] == [".pdf", ".txt"]


def test_build_payload_with_one_file_uses_a_bare_path() -> None:
    """A single path stays a string; the loader guesses the type from it."""
    payload = build_payload(
        kb_external_id="client123_project_abc",
        kb_type="auto",
        paths=["/app/data/ui_uploads/a.pdf"],
        text="",
        external_url=WEBHOOK,
    )

    assert payload.data == "/app/data/ui_uploads/a.pdf"
    assert payload.kb_type == "auto"
    assert payload.kb_external_id == "client123_project_abc"


def test_build_payload_with_several_files_uses_a_list() -> None:
    payload = build_payload(
        kb_external_id="kb",
        kb_type="vector",
        paths=["a.pdf", "b.txt"],
        text="ignored because files win",
        external_url=WEBHOOK,
    )

    assert payload.data == ["a.pdf", "b.txt"]


def test_build_payload_falls_back_to_pasted_text() -> None:
    payload = build_payload(
        kb_external_id="kb", kb_type="context", paths=[], text="  текст  ",
        external_url=WEBHOOK,
    )

    assert payload.data == "текст"


def test_build_payload_requires_some_input() -> None:
    with pytest.raises(ValueError, match="Загрузите файл"):
        build_payload(
            kb_external_id="kb", kb_type="auto", paths=[], text="   ",
            external_url=WEBHOOK,
        )


@pytest.mark.parametrize(
    ("kb_type", "expected"),
    [("auto", "auto"), ("context", "context"), ("vector", "vector")],
)
def test_requested_branch_passes_through(kb_type: str, expected: str) -> None:
    assert _requested_branch(kb_type) == expected


def test_accepted_suffixes_match_the_uploader() -> None:
    assert ACCEPTED_SUFFIXES == {".pdf", ".docx", ".txt"}


def test_the_two_kb_modes_are_distinct() -> None:
    """The mode picker branches on exactly these two labels."""
    assert KB_MODE_NEW != KB_MODE_APPEND


@pytest.mark.parametrize("branch", ["context", "vector"])
def test_append_target_problem_accepts_a_kb_on_a_known_branch(branch: str) -> None:
    assert append_target_problem({"branch": branch, "status": "ok"}) is None
    # A base whose last run failed is still a base one can extend; the UI
    # only warns about the status.
    assert append_target_problem({"branch": branch, "status": "failed"}) is None


def test_append_target_problem_reports_a_missing_kb() -> None:
    problem = append_target_problem(None)
    assert problem is not None and "не найдена" in problem


def test_append_target_problem_reports_an_unknown_branch() -> None:
    problem = append_target_problem({"branch": "auto"})
    assert problem is not None and "ветка" in problem


def test_session_loop_reuses_one_loop_for_every_coroutine() -> None:
    """The httpx pool is bound to the first loop that used it, so a second
    ``asyncio.run`` per button press would strand it on a closed loop."""
    loop = SessionLoop()
    try:
        assert loop.is_running
        first = loop.run(asyncio.sleep(0, result="a"), timeout=5)
        second = loop.run(asyncio.sleep(0, result="b"), timeout=5)

        assert (first, second) == ("a", "b")
        assert loop.is_running
    finally:
        loop._loop.call_soon_threadsafe(loop._loop.stop)
