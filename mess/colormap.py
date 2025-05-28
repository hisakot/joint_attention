import cv2

img = cv2.imread("data/test/gazecone_close/results_ds_005/000023.png")
img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
cv2.imwrite("data/test/add_000023_color.png", img)
