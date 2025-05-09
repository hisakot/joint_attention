




ffmpeg -r 1 -i data/test/pred/result/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/pred_close_kpt_lined.mp4
