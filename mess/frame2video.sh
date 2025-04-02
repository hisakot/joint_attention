ffmpeg -r 25 -start_number 44505 -i data/face_landmarks/%06d.png -vcodec libx264 -pix_fmt yuv420p -r 25 data/face_landmarks.mp4
