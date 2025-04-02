import dlib
from imutils import face_utils
import cv2

face_detector = dlib.get_frontal_face_detector()

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)

img = cv2.imread("data/test/000001.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_detector(img, 1)

for face in faces:
    landmark = face_predictor(img, face)
    landmark = face_utils.shape_to_np(naldmark)

    for (i, (x, y)) in enumerate(landmark):
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

cv2.imwrite("data/test/dlib_000001.png", img)
