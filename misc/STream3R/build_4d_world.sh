export CUDA_VISIBLE_DEVICES=1



python misc/STream3R/build_4d_world.py \
  --bg-videos final_combined_woman_room2_part1.mp4 \
  --fg-video final_combined_woman_room2_part2.mp4,final_combined_woman_room2_part3.mp4 \
  --fg-prompt "person,dog" \
  --output-dir example_imgs/4d_world_woman_room2 \
  `# Frame sampling` \
  --bg-frames "0-80" \
  --fg-frames "0-260" \
  --sample-rate 1 \
  `# STream3R inference` \
  --stream-mode window \
  --window-size 64 \
  `# Point cloud` \
  --conf-percentile 20 \
  --fg-conf-percentile 0 \
  --mask-dilate 5 \
  --bg-voxel-size 0.001 \
  `# Camera viz` \
  --frustum-size 0.04 \
  --frustum-z-push 0.15 \
  --cam-smooth 7 \
  --bg-cam-color "59,125,35" \
  --fg-cam-colors "192,79,21;33,95,154" \
  `# Render` \
  --render-width 1920 \
  --render-height 1080 \
  --render-fps 16 \
  --cam-az 10 \
  --cam-el 20 \
  --cam-dist 1.0 \
  --point-size 2.0 \
  --fov 50 \
  `# Output` \
  --sphere-radius 0.001 \
  `# --save-glb` \
  --bg-fg-extract "dog,16-64" \
  --frustum-image-scale 1.5 \
  --frustum-image-brightness 1.5 \
  `# Auto-combine` \
  --combine \
  --combine-border 12 \
  --combine-height 1200
