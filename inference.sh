#!/bin/bash

# video + audio
python inference.py --modal video_audio \
  --model_path /home/tuwenming/Models/StarJiaxing/R1-Omni-0.5B \
  --video_path ./data/test.mp4 \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
