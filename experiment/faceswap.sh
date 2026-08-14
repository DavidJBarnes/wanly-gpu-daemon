#!/usr/bin/env bash
# Locked FaceFusion stage-2 recipe (validated 2026-07-06).
# Identity-locked, expressions carried, sharp, holds through head turns/angles, NSFW-safe (body untouched).
# Uploads face+video to the 3090, runs the swap, pulls the result back here.
#
# Usage: ./faceswap.sh --face <face.jpg|png> --video <target.mp4> [--out <output.mp4>]
set -euo pipefail
HOST=david@3090.zero

FACE=""; VIDEO=""; OUT=""; FACE_INDEX="0"; DISTANCE=""; INDEX_SET=0; ALLOW_CPU=0
usage() { echo "Usage: $0 --face <face.jpg|png> --video <target.mp4> [--out <out.mp4>] [--face-index N] [--distance 0.0-1.5] [--allow-cpu]"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --face)       FACE="${2:?}"; shift 2 ;;
    --video)      VIDEO="${2:?}"; shift 2 ;;
    --out)        OUT="${2:?}"; shift 2 ;;
    --face-index) FACE_INDEX="${2:?}"; INDEX_SET=1; shift 2 ;;  # which face: 0=first,1=second (left-to-right)
    --distance)   DISTANCE="${2:?}"; shift 2 ;;                 # match looseness: high=holds angles, low=isolates one person
    --allow-cpu)  ALLOW_CPU=1; shift ;;                         # run on CPU anyway (~15x slower) instead of aborting
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1"; usage; exit 1 ;;
  esac
done
# Distance: 1.0 (loose) for single-subject so the swap holds through head turns. But at 1.0 EVERY face
# matches the reference, so a specific --face-index gets ignored and all faces swap. When a face is
# targeted, tighten to 0.6 so only that person matches (isolates them). Override with --distance.
if [ -z "$DISTANCE" ]; then [ "$INDEX_SET" = "1" ] && DISTANCE="0.6" || DISTANCE="1.0"; fi

[ -n "$FACE" ]  || { echo "ERROR: --face required";  usage; exit 1; }
[ -n "$VIDEO" ] || { echo "ERROR: --video required"; usage; exit 1; }
[ -f "$FACE" ]  || { echo "ERROR: face not found: $FACE";   exit 1; }
[ -f "$VIDEO" ] || { echo "ERROR: video not found: $VIDEO"; exit 1; }
OUT="${OUT:-swapped_$(basename "${VIDEO%.*}").mp4}"

RSRC="/tmp/ff_src_$$.${FACE##*.}"; RTGT="/tmp/ff_tgt_$$.mp4"; ROUT="/tmp/ff_out_$$.mp4"

FF_PY="~/miniconda3/envs/facefusion/bin/python"
FF_PIP="~/miniconda3/envs/facefusion/bin/pip"

# Ask onnxruntime what it can actually do BEFORE uploading anything.
#
# On 2026-08-11 a bare `onnxruntime` (CPU-only) wheel got installed on top of `onnxruntime-gpu` in
# this env. They unpack into the SAME package dir, the CPU .so files win, and CUDAExecutionProvider
# vanishes -- at which point FaceFusion drops `cuda` from its argparse choices and this script died
# on "invalid choice: 'cuda' (choose from 'cpu')", which says nothing about the real cause. The
# fallback that matters is silent elsewhere (the RunPod path just runs on CPU at GPU prices), so
# the rule here is: name the cause, and never quietly spend 15x the wall clock.
echo "[1/5] checking the execution provider on ${HOST#*@}..."
PROVIDERS="$(ssh "$HOST" "$FF_PY -c \"import onnxruntime as ort; print(','.join(ort.get_available_providers()))\"" 2>/dev/null || true)"

if printf '%s' "$PROVIDERS" | grep -q 'CUDAExecutionProvider'; then
  EP="cuda"
  echo "      CUDAExecutionProvider present."
else
  EP="cpu"
  echo
  if [ -z "$PROVIDERS" ]; then
    echo "  !! Could not query onnxruntime in the facefusion env on $HOST."
    echo "     Either the host is unreachable or the env is gone. Providers unknown."
  else
    echo "  !! No CUDAExecutionProvider. onnxruntime reports: $PROVIDERS"
    # The shadowing is the known cause, so check for it by name rather than making the operator
    # rediscover it. Both packages present == this exact bug (wanly-gpu-daemon#135).
    INSTALLED="$(ssh "$HOST" "$FF_PIP list 2>/dev/null | grep -iE '^onnxruntime(-gpu)?[[:space:]]'" 2>/dev/null || true)"
    [ -n "$INSTALLED" ] && { echo "     Installed:"; printf '%s\n' "$INSTALLED" | sed 's/^/       /'; }
    if printf '%s' "$INSTALLED" | grep -qiE '^onnxruntime[[:space:]]' && \
       printf '%s' "$INSTALLED" | grep -qiE '^onnxruntime-gpu[[:space:]]'; then
      echo "     ^ BOTH are installed. The CPU build is shadowing the GPU one -- this is #135."
    fi
    echo
    echo "     Repair (facefusion 3.2.0 pins 1.21.1):"
    echo "       ssh $HOST '$FF_PIP uninstall -y onnxruntime onnxruntime-gpu && $FF_PIP install onnxruntime-gpu==1.21.1'"
    echo "       ssh $HOST '$FF_PY -c \"import onnxruntime as o; print(o.get_available_providers())\"'"
  fi
  echo
  echo "     CPU still produces correct output, just ~15x slower (~500-870s/clip vs ~37s)."
  if [ "$ALLOW_CPU" != "1" ]; then
    echo "     Refusing to run on CPU by default. Re-run with --allow-cpu if that is what you want."
    exit 1
  fi
  echo "     --allow-cpu given; continuing on CPU."
  echo
fi

echo "[2/5] uploading face + video..."
scp -q "$FACE" "$HOST:$RSRC"; scp -q "$VIDEO" "$HOST:$RTGT"

echo "[3/5] freeing GPU + running FaceFusion (provider $EP, occlusion mask ON, face-index $FACE_INDEX, distance $DISTANCE)..."
ssh "$HOST" "curl -sf -X POST http://localhost:8188/free -H 'Content-Type: application/json' -d '{\"unload_models\":true,\"free_memory\":true}' -o /dev/null 2>/dev/null; \
  cd ~/projects/facefusion && ~/miniconda3/envs/facefusion/bin/python facefusion.py headless-run \
  -s '$RSRC' -t '$RTGT' -o '$ROUT' \
  --processors face_swapper face_enhancer \
  --face-swapper-model inswapper_128 --face-swapper-pixel-boost 512x512 \
  --face-enhancer-model gfpgan_1.4 \
  --face-mask-types box occlusion \
  --face-selector-mode reference --reference-face-distance $DISTANCE \
  --face-selector-order left-right --reference-face-position $FACE_INDEX \
  --execution-providers $EP 2>&1 | tail -2"

echo "[4/5] pulling result..."
scp -q "$HOST:$ROUT" "$OUT"
ssh "$HOST" "rm -f '$RSRC' '$RTGT' '$ROUT'" 2>/dev/null || true
echo "[5/5] DONE -> $OUT"
