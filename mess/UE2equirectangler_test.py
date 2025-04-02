import json
import math

import cv2
import numpy as np

H = 1920
W = 3840
F = 420 # frame

img = cv2.imread("data/MovieRenders/LevelSequence_OR_0420.png")

camera = {"x" : 600, "y" : 450, "z" : 170}

def transform(cod_3d, camera):
    cod_3d = np.array([cod_3d["x"], cod_3d["y"], cod_3d["z"]])
    camera = np.array([camera["x"], camera["y"], camera["z"]])
    camera_cod = cod_3d - camera

    length = np.linalg.norm(camera_cod, ord=2)
    camera_cod /= length

    theta = math.asin(camera_cod[2])
    phi = math.asin(camera_cod[1] / math.cos(theta))

    x_norm = phi / math.pi
    y_norm = 2 * theta / math.pi

    x_c = - x_norm * W / 2
    y_c = - y_norm * H / 2

    x = x_c + W / 2
    y = y_c + H / 2

    return x, y

with open("data/UE_json/Aoi_face.json") as f:
    data = json.load(f)
    print(len(data["Structure_face"]))

    cod_3d = data["Structure_face"][F]["nose"]
    x, y = transform(cod_3d, camera)
    cv2.circle(img, (int(x), int(y)), 10, (255, 0, 255), 2)
    cv2.putText(img, "f0", (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    cv2.putText(img, "f0: nose", (100, 100 + 18 * 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))


with open("data/UE_json/Aoi_body.json") as f:
    data = json.load(f)
    print(len(data["Structure_body"]))
    for i, part in enumerate(data["Structure_body"][F]):
        value = data["Structure_body"][F][part]
        x, y = transform(value, camera)
        cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), 2)
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        cv2.putText(img, str(i) + ": " + part, (100, 100 + i * 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))

with open("data/UE_json/Bernice_body.json") as f:
    data = json.load(f)
    print(len(data["Structure_body"]))
    for i, part in enumerate(data["Structure_body"][F]):
        value = data["Structure_body"][F][part]
        x, y = transform(value, camera)
        cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 2)
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

with open("data/UE_json/Danielle_body.json") as f:
    data = json.load(f)
    print(len(data["Structure_body"]))
    for i, part in enumerate(data["Structure_body"][F]):
        value = data["Structure_body"][F][part]
        x, y = transform(value, camera)
        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 2)
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

cv2.imwrite("./test_img.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


