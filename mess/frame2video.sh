ffmpeg -r 1 -i data/test/pred/result/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/swinunet_2ch_cossim_MSE_0.8.mp4
