import tflite_runtime.interpreter as tflite
from PIL import Image
import datetime
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def process_outs(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    print(prediction.shape)
    print("10100100101001010010010100101000101010100101010")
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    print(nc, "THIS IS NC")
    for x in xc[0]:
        print(x, "asdfasdf")
    return

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
    outs = np.array(outs)
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
        print(outs.shape)
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
