import cv2

img = cv2.imread("data/test/gazecone_body/results_ds_005/000004.png")
img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
cv2.imwrite("data/test/000004.png", img)
