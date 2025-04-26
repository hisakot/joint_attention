ffmpeg -r 1 -i data/test/pred/img_gazecone_nch_cossim/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/pred_nch_cossim.mp4
