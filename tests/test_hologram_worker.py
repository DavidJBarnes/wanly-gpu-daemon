"""The hologram-only worker in the 2070's scene container (wanly-gpu-docker#228).

It claims against the row the container's supervisor registered, never its own, and only when
the container has memory headroom -- RAM is what it shares with the captioner.
"""
import uuid

import pytest

from daemon import hologram_worker as hw

WID = uuid.UUID("11111111-2222-3333-4444-555555555555")


class FakeQueue:
    def __init__(self, segment=None):
        self.segment = segment
        self.claims = []
        self.updates = []

    async def claim_next(self, worker_id, name, kind=None, pending_loras=None):
        self.claims.append((worker_id, kind))
        seg, self.segment = self.segment, None
        return seg

    async def update_segment(self, segment_id, result):
        self.updates.append((segment_id, result))


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A worker-id file and a cgroup directory: 8 GiB cap, 2 GiB anonymous."""
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{8 * 1024 ** 3}\n")
    (cg / "memory.stat").write_text(f"anon {2 * 1024 ** 3}\nfile {5 * 1024 ** 3}\n")
    idf = tmp_path / "worker-id"
    idf.write_text(str(WID))
    monkeypatch.setattr(hw, "CGROUP", str(cg))
    monkeypatch.setattr(hw, "WORKER_ID_FILE", str(idf))
    monkeypatch.setattr(hw, "MIN_HEADROOM_GB", 3.0)
    return cg, idf


def test_headroom_counts_anonymous_memory_not_page_cache(box):
    # 8 - 2 = 6, although anon + file is 7 of 8.
    assert hw.headroom_gb() == pytest.approx(6.0)


def test_uncapped_container_has_no_headroom_limit(box):
    cg, _ = box
    (cg / "memory.max").write_text("max\n")
    assert hw.headroom_gb() is None
    assert hw.may_claim() == (True, "")


async def test_claims_holograms_against_the_supervisors_row(box, monkeypatch):
    ran = []

    async def execute(segment, comfyui, queue):
        ran.append((segment, comfyui))

    monkeypatch.setattr(hw, "execute_segment", execute)
    q = FakeQueue(segment="seg")
    assert await hw.poll_once(q) is True
    assert q.claims == [(WID, "hologram")]
    assert ran == [("seg", None)]                # no ComfyUI


async def test_waits_for_registration(box):
    _, idf = box
    idf.unlink()
    q = FakeQueue(segment="seg")
    assert await hw.poll_once(q) is False
    assert q.claims == []


async def test_waits_for_memory_headroom(box):
    cg, _ = box
    (cg / "memory.stat").write_text(f"anon {int(5.5 * 1024 ** 3)}\n")
    assert hw.may_claim()[0] is False
    q = FakeQueue(segment="seg")
    assert await hw.poll_once(q) is False
    assert q.claims == []


async def test_a_crash_is_reported_as_a_failed_segment(box, monkeypatch):
    async def execute(segment, comfyui, queue):
        raise RuntimeError("boom")

    monkeypatch.setattr(hw, "execute_segment", execute)
    seg = type("Seg", (), {"id": "abc"})()
    q = FakeQueue(segment=seg)
    assert await hw.poll_once(q) is True
    [(sid, result)] = q.updates
    assert sid == "abc" and result.status == "failed" and "boom" in result.error_message


def test_announces_ready_for_the_supervisor(tmp_path, monkeypatch):
    path = tmp_path / "run" / "hologram.ready"
    monkeypatch.setenv("WANLY_READY_FILE", str(path))
    hw.announce_ready()
    assert path.read_text() == "ready\n"
