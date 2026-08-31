"""The PainterLongVideo swap is gone, and must not come back (wanly-gpu-daemon#124).

It replaced node 98 with PainterLongVideo whenever a segment had both an initial reference image
and a start image. On Wan 2.2 i2v every distinguishing input was dead:

  previous_video      wired to a single-frame LoadImage, so the motion reference resolved to None
  reference_latents   only read by checkpoints carrying ref_conv.weight (Wan 2.1 FLF2V)
  clip_vision_output  Wan 2.2 i2v has no img_emb; the official template has no CLIPVisionLoader
  motion_amplitude    applied only on the previous-video-only branch, which this graph never took
  end_image           never passed

With start and end connected the node does line for line what stock WanFirstLastFrameToVideo
does, so it was WanImageToVideo with a wasted VAE encode and a wasted CLIP Vision load -- plus a
"Segment N motion_amplitude: X" log line for a value that was computed and discarded, which is
what made it look like a live tuning lever during the motion investigation.

The reference-image parameters are still ACCEPTED, because segments queued with an
identity_reference_image must keep building.
"""

from tests.conftest import make_segment

from daemon.workflow_builder import build_workflow


class TestSwapIsGone:
    def test_reference_image_still_builds_the_stock_node(self):
        wf = build_workflow(
            make_segment(index=1),
            start_image_filename="start.png",
            initial_reference_image_filename="original.png",
        )
        assert wf["98"]["class_type"] == "WanImageToVideo"

    def test_no_clip_vision_nodes_are_added(self):
        # Loading a CLIP Vision model to feed an input nothing reads is pure cost on a card
        # already tight for VRAM.
        wf = build_workflow(
            make_segment(index=1),
            start_image_filename="start.png",
            initial_reference_image_filename="original.png",
        )
        classes = {n.get("class_type") for n in wf.values()}
        assert "CLIPVisionLoader" not in classes
        assert "CLIPVisionEncode" not in classes
        assert "PainterLongVideo" not in classes

    def test_a_reference_image_changes_nothing_at_all(self):
        # The strongest statement of the finding: passing the reference produced an identical
        # graph, so every run that took this path paid for it and got nothing.
        common = dict(start_image_filename="start.png")
        with_ref = build_workflow(make_segment(index=1),
                                  initial_reference_image_filename="original.png", **common)
        without = build_workflow(make_segment(index=1), **common)
        assert with_ref == without

    def test_previous_motion_magnitude_is_accepted_and_inert(self):
        # It only ever fed motion_amplitude on a branch this graph never took.
        a = build_workflow(make_segment(index=1), start_image_filename="s.png",
                           previous_motion_magnitude=0.9)
        b = build_workflow(make_segment(index=1), start_image_filename="s.png")
        assert a == b
