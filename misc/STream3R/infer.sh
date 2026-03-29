export CUDA_VISIBLE_DEVICES=1

# Example: video with sample rate
# python misc/STream3R/infer_stream3r.py \
#   example_imgs/final_video.mp4 \
#   --sample-rate 54 \
#   -o outputs/flamingo.ply \
#   --no-shuffle

# Example: video with specific frames (0,1,2,3-10,15 = frames 0,1,2,3,4,5,6,7,8,9,10,15)
python misc/STream3R/infer_stream3r.py \
  example_imgs/final_video.mp4 \
  --frames "0-65" \
  --sample-rate 4 \
  -o outputs/flamingo.ply \
  --no-shuffle

# Example: single image
# python misc/STream3R/infer_stream3r.py \
#   example_imgs/final.png example_imgs/dog_bowl_mid_res.png \
#   -o outputs/final2.ply
  
# python misc/STream3R/infer_stream3r.py \
#   example_imgs/dowbowl_v2.png \
#   -o outputs/dowbowl_v2.ply
