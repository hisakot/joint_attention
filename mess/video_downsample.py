import argparse
import os

import cv2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--sample", required=True, type=int)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("file was not able to open.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_frame_dir = os.path.join(args.save_dir, "frame")
    if not os.path.exists(save_frame_dir):
        os.mkdir(save_frame_dir)

    output_file = os.path.join(args.save_dir, "downsampled_" + os.path.basename(args.video))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), isColor=True)

    frame_num = 0
    with tqdm(total=total_frame_num) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                if frame_num % args.sample == 0:
                    frame_path = os.path.join(save_frame_dir, str(frame_num).zfill(6) + ".png")
                    cv2.imwrite(frame_path, frame)
                    video_writer.write(frame)
            else:
                break
            frame_num += 1
            pbar.update(1)

    cap.release()
    video_writer.release()


if __name__ == '__main__':
    main()

