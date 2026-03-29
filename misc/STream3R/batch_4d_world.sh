#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

VIDEOS=(
  "more_results/corgi.mp4"
  "more_results/horse_2.mp4"
  "more_results/horse.mp4"
  "more_results/horses.mp4"
  "more_results/husky.mp4"
  "more_results/pilot.mp4"
  "more_results/squirral.mp4"
  "more_results/woman.mp4"
)

for VID in "${VIDEOS[@]}"; do
  # Extract base name without extension: e.g. "corgi", "horse_2"
  BASENAME=$(basename "$VID" .mp4)
  echo "============================================================"
  echo "Processing: $BASENAME"
  echo "============================================================"

  # Get video dimensions and frame count
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$VID")
  H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VID")
  NFRAMES=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$VID")
  HALF_W=$((W / 2))

  # Split into part1 (left=Renderer) and part2 (right=Monitor)
  PART1="more_results/${BASENAME}_part1.mp4"
  PART2="more_results/${BASENAME}_part2.mp4"

  if [ ! -f "$PART1" ] || [ ! -f "$PART2" ]; then
    echo "  Splitting $VID -> part1 (${HALF_W}x${H}) + part2 (${HALF_W}x${H})"
    ffmpeg -y -i "$VID" -filter:v "crop=${HALF_W}:${H}:0:0" -c:v libx264 -pix_fmt yuv420p -crf 18 -r 16 "$PART1"
    ffmpeg -y -i "$VID" -filter:v "crop=${HALF_W}:${H}:${HALF_W}:0" -c:v libx264 -pix_fmt yuv420p -crf 18 -r 16 "$PART2"
  else
    echo "  Parts already exist, skipping split"
  fi

  # fg-prompt: map filename to SAM3-recognizable object name
  case "$BASENAME" in
    corgi)     FG_PROMPT="dog" ;;
    horse_2)   FG_PROMPT="horse" ;;
    horse)     FG_PROMPT="horse" ;;
    horses)    FG_PROMPT="horse" ;;
    husky)     FG_PROMPT="dog" ;;
    pilot)     FG_PROMPT="person" ;;
    squirral)  FG_PROMPT="squirrel" ;;
    woman)     FG_PROMPT="person" ;;
    *)         FG_PROMPT="$BASENAME" ;;
  esac

  # bg-frames: horse_2 uses 150, others use 80
  if [ "$BASENAME" = "horse_2" ]; then
    BG_FRAMES="0-150"
  else
    BG_FRAMES="0-80"
  fi

  # fg-frames: use total frame count
  FG_FRAMES="0-${NFRAMES}"

  OUTPUT_DIR="more_results/finised/4d_world_${BASENAME}"

  echo "  fg-prompt: $FG_PROMPT"
  echo "  bg-frames: $BG_FRAMES, fg-frames: $FG_FRAMES"
  echo "  output: $OUTPUT_DIR"

  # Skip if already completed
  if [ -f "$OUTPUT_DIR/4d_world_combined.mp4" ]; then
    echo "  Already done, skipping"
    continue
  fi

  python misc/STream3R/build_4d_world.py \
    --bg-videos "$PART1" \
    --fg-video "$PART2" \
    --fg-prompt "$FG_PROMPT" \
    --output-dir "$OUTPUT_DIR" \
    --bg-frames "$BG_FRAMES" \
    --fg-frames "$FG_FRAMES" \
    --sample-rate 1 \
    --stream-mode window \
    --window-size 64 \
    --conf-percentile 20 \
    --fg-conf-percentile 0 \
    --mask-dilate 5 \
    --bg-voxel-size 0.001 \
    --frustum-size 0.04 \
    --frustum-z-push 0.15 \
    --cam-smooth 7 \
    --bg-cam-color "59,125,35" \
    --fg-cam-colors "192,79,21" \
    --render-width 1920 \
    --render-height 1080 \
    --render-fps 16 \
    --cam-az 10 \
    --cam-el 20 \
    --cam-dist 1.0 \
    --point-size 2.0 \
    --fov 50 \
    --sphere-radius 0.001 \
    --frustum-image-scale 1.5 \
    --frustum-image-brightness 1.5 \
    --combine \
    --combine-border 12 \
    --combine-height 1200

  echo "  Done: $BASENAME"
  echo ""
done

echo "All videos processed!"
