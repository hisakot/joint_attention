ffmpeg -r 1 -i data/test/pred/kpt_gaze/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/kpt_gaze_body.mp4
