# experiment/

Operator tools that drive the 3090 by hand, outside the daemon's job loop. Nothing here is
imported by `daemon/` — these are scripts run from a laptop when a recipe needs to be applied to
specific files rather than to a queued job.

## faceswap.sh

The locked FaceFusion stage-2 recipe, validated 2026-07-06: identity-locked, expressions carried,
holds through head turns, body untouched (so it is safe on NSFW output). It uploads a face and a
target video to 3090.zero, runs the swap there, and pulls the result back.

    ./faceswap.sh --face k3llydw.png --video segment.mp4 --out swapped.mp4

The settings in the script are the recipe — `--reference-face-distance` and the default landmarker
are what make identity hold through angle changes, so treat the flags as findings rather than
defaults to tune casually. `--distance` and `--face-index` exist for the multi-subject case: at the
default 1.0 every face matches the reference, so targeting one person requires tightening to ~0.6.

### It depends on an env this repo does not own

The swap runs in the `facefusion` conda env on 3090.zero, which is a separate machine-local
install. That env broke once already: a bare `onnxruntime` (CPU) wheel installed on top of
`onnxruntime-gpu` shadowed it, CUDA silently disappeared, and the script died on an argparse error
that named nothing useful (wanly-gpu-daemon#135). Upstream facefusion's own `requirements.txt`
pins the CPU wheel, so a reinstall can reintroduce it at any time (#139).

The script therefore checks `onnxruntime.get_available_providers()` before uploading anything and
aborts with the repair command if CUDA is missing. `--allow-cpu` overrides that — correct output,
roughly 15x slower (~500-870s/clip vs ~37s), so it is opt-in rather than a silent fallback.

FaceFusion also wants the GPU relatively free; with a generation holding ~19GB it will OOM on a
small allocation. Run bulk swaps against an idle daemon.
