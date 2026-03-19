# ffmpeg -r 30 -i data/ue/01/MovieRenders/%04d.png -vcodec libx264 -pix_fmt yuv420p -r 30 data/ue/ue_01.mp4
# ffmpeg -r 10 -i data/test/pred/result_color/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 10 data/test/pred/ue_train_real_test_cnn.mp4
# ffmpeg -r 1 -i ~/Lab/PJAE-ICCV2023-UE/results/Medical/with_ann_action_50epoch/final_jo_att_superimposed/test_ds_005_000001_%06d_final_jo_att_superimposed.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/pred/PJAE_with_ann_50epoch.mp4
# ffmpeg -r 30 -i data/UE/MovieRenders/%04d.jpeg -vcodec libx264 -pix_fmt yuv420p -r 30 data/UE/ue_001.mp4
ffmpeg -r 1 -i data/ue/train/frames/ds_ue_01/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 3 data/ue/ds_ue_01.mp4
