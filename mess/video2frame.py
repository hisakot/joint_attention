import cv2
import os
from tqdm import tqdm


def change_size(image):
 
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    
    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left  
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom  

    pre1_picture = image[left:left + width, bottom:bottom + height]  

    #print(pre1_picture.shape) 
    
    return pre1_picture  


def save_all_frames(video_path, dir_path, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    with tqdm(total=44980) as pbar:
        while True:
            ret, frame = cap.read()
            # dim = (int(frame.shape[1]/frame.shape[0]*300), 300)
            # frame = cv2.resize(frame, dim)
            # frame = change_size(frame)
            # frame = cv2.resize(frame, (480, 270))
#frame = cv2.resize(frame, (1920, 1080))
            if ret:
                cv2.imwrite('{}{}.{}'.format(dir_path, str(n+1).zfill(6), ext), frame)
                n += 1
                pbar.update()
            else:
                return

def save_a_part_of_frames(video_path, dir_path, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    i = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if n % 25 == 0:
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                cv2.imwrite('{}{}.{}'.format(dir_path, str(i).zfill(6), ext), frame)
                i += 1
            n += 1
        else:
            return
# save_all_frames('E:/2022-04-22DrHayashida/fullstream2.mp4', 'E:/2022-04-22DrHayashida/org_imgs/', 'png')
# save_all_frames('../insta360Go3/Tobii/fullstream.mp4', '../insta360Go3/Tobii/fps_imgs/', 'png')
# save_all_frames('../insta360Go3/004_withTobii.mp4', '../insta360Go3/fps_imgs/', 'png')
# save_a_part_of_frames('/mnt/g/x4_data/x4_360_004.mp4', '../Saliency/images/', 'png')
save_all_frames('data/video_009.mp4', 'data/frames/', 'png')

