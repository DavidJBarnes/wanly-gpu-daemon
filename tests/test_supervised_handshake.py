"""What the daemon says to a supervisor that runs it as a child (wanly-gpu-docker#83).

One container per GPU: the daemon has no port, so readiness is a file it writes after
registering; the trainer beside it claims under the same worker row, so the id is a file too;
and the supervisor says what the box is (kinds) and runs (provides) through the environment.
Outside the supervisor none of the variables are set and nothing changes.
"""
import inspect
import os

from daemon import main as mod
from daemon import queue_client as qc


def test_the_ready_and_id_files_are_written_after_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("WANLY_READY_FILE", str(tmp_path / "run" / "daemon.ready"))
    monkeypatch.setenv("WORKER_ID_FILE", str(tmp_path / "run" / "worker-id"))
    mod._announce_registered("11111111-2222-3333-4444-555555555555")
    assert (tmp_path / "run" / "daemon.ready").read_text().strip() == "ready"
    assert (tmp_path / "run" / "worker-id").read_text().strip() == "11111111-2222-3333-4444-555555555555"


def test_it_happens_right_after_the_registration_returns():
    src = inspect.getsource(mod.run)
    assert src.index("friendly_name_ref = [registered_name]") < src.index("_announce_registered(worker_id)")


def test_nothing_is_written_outside_the_supervisor(tmp_path, monkeypatch):
    monkeypatch.delenv("WANLY_READY_FILE", raising=False)
    monkeypatch.delenv("WORKER_ID_FILE", raising=False)
    mod._announce_registered("x")
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_path_is_a_warning_not_a_crash(monkeypatch):
    monkeypatch.setenv("WANLY_READY_FILE", "/proc/nope/daemon.ready")
    mod._announce_registered("x")   # must not raise


def test_kinds_and_provides_come_from_the_environment():
    src = inspect.getsource(qc.QueueClient.register)
    assert 'os.environ.get("WORKER_KINDS"' in src and 'os.environ.get("WORKER_PROVIDES"' in src
    assert 'payload["kinds"] = kinds' in src and 'payload["provides"] = provides' in src
