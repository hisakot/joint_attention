import cv2

img = cv2.imread("data/train/frames/ds014/000005.png")

def generate_saliency(img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliency_map) = saliency.computeSaliency(img)
    if success is False:
        return (False, None)
    saliency_map = (saliency_map * 255).astype("uint8")
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    weight = cv2.addWeighted(img, 0.7, heatmap, 0.5, 1.0)
    return (success, weight)

def generate_video_saliency(video_path, output_path):
    input_video = cv2.VideoCapture(video_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while True:
        ret, frame = input_video.read()
        if ret is not True:
            break
        (success, saliency_map) = generate_saliency(frame)
        if success is True:
            output_video.write(saliency_map)
    input_video.release()
    output_video.release()

(success, saliency_map) = generate_saliency(img)
if success is True:
    saliency_map = cv2.resize(saliency_map, None, fx=0.5, fy=0.5)
    cv2.imshow("saliency", saliency_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

generate_video_saliency("data/train/ds_014.mp4", "mess/saliency.mp4")
