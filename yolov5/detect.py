import tflite_runtime.interpreter as tflite
from PIL import Image
import datetime
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import cv2
from general import process_outs, process_image, get_data_dict, scale_coords, Annotator, colors

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def detect_image(image, interpreter, imgsz, data, pathname):
    pimage = process_image(image,imgsz)
    im0 = np.array(image)
    img = pimage
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, np.array(pimage, dtype="uint8"))
    start = datetime.datetime.now()
    interpreter.invoke()
    time = datetime.datetime.now() - start
    outs = interpreter.get_tensor(output_index)
    outs = [np.array(outs)]
    pred = process_outs(outs[0])
    results = np.unique(pred[:,5], return_counts=True)
    results = ([(data[int(i)]+"s") for i in results[0]], results[1])
    result_s = "Found: "
    for x in range(0,len(results[0])):
        if x != len(results[0])-1:
            result_s+=str(int(results[1][x])) + " " + results[0][x] + ", "
        else:
            result_s+=str(int(results[1][x])) + " " + results[0][x]
    print(result_s)
    print(time)

    print(pred)
    for i, det in enumerate(pred):
        print(i,det)
        print(img.shape)
        print(im0.shape)
        print(img.shape[1:2])
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        annotator = Annotator(im0, line_width=3, example=str(data))
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = data[c]
            annotator.box_label(xyxy, label, color=colors(c, True))
        im0 = annotator.result()
        if True:
                cv2.imwrite("test"+pathname, im0)
    return outs, time


def detect_video(video, interpreter, imgsz, data):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    time_array = []
    while success and count<20:
        outs, time = detect_image(Image.fromarray(image), interpreter, imgsz, data)
        time_array.append(time)
        success, image = vidcap.read()
        count += 1
    print(np.mean(np.array(time_array)))
    return


def run(weights=ROOT / 'yolov5s.pt', source=ROOT / 'data/images', imgsz=256, data="datasets/LPCV.yaml"):
    model_path, source, imgsz, data= opt.weights, opt.source, opt.imgsz, opt.data
    data = get_data_dict(data)['names']
    interpreter = tflite.Interpreter(model_path)
    interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
    # import tensorflow as tf

    # interpreter = tf.lite.Interpreter(model_path)
    # print(type(interpreter))

    interpreter.allocate_tensors()
    
    if source.endswith("jpg") or source.endswith("jpeg"):
        image = Image.open(source)
        detect_image(image, interpreter, imgsz, data, source)
    elif source.endswith("m4v") or source.endswith("mp4"): 
        video = source
        detect_video(video, interpreter, imgsz, data)
    else:
        return


def main(opt):
    run(**vars(opt))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "./models/test.tflite", help="model path(s)",)
    parser.add_argument("--source", type=str,default=ROOT / "./sources/test.jpg",)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument('--data', type=str, default=ROOT / 'datasets/LPCV.yaml', help='dataset.yaml path')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
