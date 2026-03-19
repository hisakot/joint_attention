ffmpeg -r 30 -i data/ue/01/MovieRenders/%04d.png -vcodec libx264 -pix_fmt yuv420p -r 30 data/ue/ue_01.mp4
# ffmpeg -r 1 -i ~/Lab/PJAE-ICCV2023-UE/results/Medical/with_ann_action_50epoch/final_jo_att_superimposed/test_ds_005_000001_%06d_final_jo_att_superimposed.png -vcodec libx264 -pix_fmt yuv420p -r 1 data/test/pred/PJAE_with_ann_50epoch.mp4
