import cv2
import glob

img_paths = glob.glob("data/test/pred/result1/*.png")
for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite("data/test/pred/result1_colormap/" + str(i).zfill(6) + ".png", img)
