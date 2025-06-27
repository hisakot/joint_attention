ffmpeg -r 1 -i data/test/pred/result2/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/swin_t_cossim.mp4
