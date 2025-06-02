ffmpeg -r 1 -i data/test/pred/result2/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/selected_augmentation.mp4
