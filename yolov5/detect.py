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


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[int(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[int(cl)], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


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
    scale, zero_point = interpreter.get_output_details()[0]["quantization"]
    pred = (outs[0].astype(np.float32) - zero_point) * scale  # re-scale
    print(pred)
    pred[..., 0] *= imgsz  # x
    pred[..., 1] *= imgsz  # y
    pred[..., 2] *= imgsz  # w
    pred[..., 3] *= imgsz  # h
    print(pred)
    pred = process_outs(pred)
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
    
    boxes = pred[:,:4]
    image = cv2.imread(pathname)
    shape = image.shape
    width, height = shape[1]/256, shape[0]/256
    image_dims = [width, height, width/2, height/2]
    boxes = boxes * image_dims
    scores = pred[:,4]
    classes = pred[:,5]
    if boxes is not None:
        draw(image, boxes, scores, classes, data)
    return image


    # print(pred)
    # det = pred
    # print(img.shape)
    # print(im0.shape)
    # print(img.shape[1:3])
    # print(det)
    # print(det[:,:4])
    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    # print(det)
    # annotator = Annotator(im0, line_width=3, example=str(data))
    # for *xyxy, conf, cls in reversed(det):
    #     c = int(cls)  # integer class
    #     label = data[c]
    #     print(xyxy)
    #     annotator.box_label(xyxy, label, color=colors(c, True))
    # im0 = annotator.result()
    # if True:
    #     print("asdfas")
    #     cv2.imwrite(pathname[len(pathname)-5:]+"test.jpg", im0)
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
        new_image = detect_image(image, interpreter, imgsz, data, source)
        cv2.imwrite(source[:len(source)-5]+"456test.jpg", new_image)
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
