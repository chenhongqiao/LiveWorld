export CUDA_VISIBLE_DEVICES=1

python misc/STream3R/infer_stream3r_4d.py \
  --bg-video final_combined_kids_coffee_part1.mp4 \
  --fg-video final_combined_kids_coffee_part2.mp4,final_combined_kids_coffee_part3.mp4 \
  --fg-prompt "person,dog" \
  --output-dir example_imgs/4d_kids_coffee \
  `# Frame sampling` \
  --sample-rate 8 \
  --bg-frames "0-80" \
  --save-frames "0-260" \
  `# STream3R inference` \
  --stream-mode window \
  --window-size 32 \
  --conf-percentile 20 \
  --fg-conf-percentile 0 \
  `# Foreground segmentation` \
  --mask-dilate 5 \
  `# Point cloud output` \
  --bg-voxel-size 0.001 \
  --sphere-radius 0.001 \
  --no-ply \
  --save-combined-glb \
  `# Render settings` \
  --render-fps 8 \
  --cam-az 0 \
  --cam-el 0 \
  --cam-dist 1 \
  --point-size 1.0 \
  --fov 50
