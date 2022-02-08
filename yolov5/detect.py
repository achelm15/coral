import tflite_runtime.interpreter as tflite
from PIL import Image
import datetime
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import cv2
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def process_outs(prediction, conf_thres=25, iou_thres=45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    print(prediction.shape)
    print("10100100101001010010010100101000101010100101010")
    print(prediction)
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = np.zeros((0,6))*prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = np.concatenate((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]


        # Check shapes
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        print(boxes.shapeks, scores.shape)
        # print(boxes.shape, scores.shape, "999999999999999999999999999999999999")
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # if i.shape[0] > max_det:  # limit detections
        #     i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        # output[xi] = x[i]
        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded



    return

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, np.array) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process_image(img, imgsz):
    image = np.array(img.resize((imgsz, imgsz)), dtype="int8")
    #     image /= 255.
    image = np.expand_dims(image, axis=0)
    return image


def detect_image(image, interpreter, imgsz):
    pimage = process_image(image,imgsz)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, np.array(pimage, dtype="uint8"))
    start = datetime.datetime.now()
    interpreter.invoke()
    time = datetime.datetime.now() - start
    outs = interpreter.get_tensor(output_index)
    shape = np.array(pimage).shape
    outs = [np.array(outs)]
    image = np.array(image)
    process_outs(outs[0])
    print(time)
    return outs, time


def detect_video(video, interpreter, imgsz):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    time_array = []
    while success and count<20:
        outs, time = detect_image(Image.fromarray(image), interpreter, imgsz)
        print(np.array(outs).shape)
        time_array.append(time)
        success, image = vidcap.read()
        count += 1
    print(np.mean(np.array(time_array)))
    return


def run(weights=ROOT / 'yolov5s.pt', source=ROOT / 'data/images', imgsz=256):
    model_path = opt.weights
    source = opt.source
    imgsz = opt.imgsz
    interpreter = tflite.Interpreter(model_path)
    interpreter = tflite.Interpreter(
        model_path, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
    )
    # import tensorflow as tf

    # interpreter = tf.lite.Interpreter(model_path)
    # print(type(interpreter))

    interpreter.allocate_tensors()
    
    if source.endswith("jpg") or source.endswith("jpeg"):
        source = Image.open(source)
        detect_image(source, interpreter, imgsz)
    elif source.endswith("m4v") or source.endswith("mp4"): 
        video = source
        detect_video(video, interpreter, imgsz)
    else:
        return


def main(opt):
    run(**vars(opt))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "./models/test.tflite", help="model path(s)",)
    parser.add_argument("--source", type=str,default=ROOT / "./sources/test.jpg",)
    parser.add_argument("--imgsz", type=int, default=256)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
