ffmpeg -r 1 -i data/test/pred/no_data_augmentation/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/no_augmentation_gray.mp4
