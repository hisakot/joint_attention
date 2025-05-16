ffmpeg -r 1 -i data/test/pred/result2/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/no_data_augumentation_color.mp4
