ffmpeg -r 1 -i data/test/pred/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/pred_cossim.mp4
