import tflite_runtime.interpreter as tflite
from PIL import Image
import datetime
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import cv2
from general import process_outs, get_data_dict, scale_coords, letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def draw(image, boxes, scores, classes, all_classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        top = max(0, np.floor(x1 + 0.5).astype(int))
        left = max(0, np.floor(y1 + 0.5).astype(int))
        right = max(0, np.floor(x2 + 0.5).astype(int))
        bottom = max(0, np.floor(y2 + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[int(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print(' box coordinate x,y,w,h: {0} '.format(box) + 'class: {0}, score: {1:.2f}'.format(all_classes[int(cl)], score) )


def detect_image(image, interpreter, imgsz, data, pathname, conf):
    if pathname:
        test_img0 = cv2.imread(pathname)
    else: 
        test_img0 = np.array(image)
    test_img = letterbox(test_img0, imgsz, stride=64, auto=False)[0]
    test_img = test_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    test_img = np.ascontiguousarray(test_img)
    _, h, w = test_img.shape
    test_img = test_img.astype("float64")
    test_img /= 255
    input, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero_point = input['quantization']
    test_img = (test_img / scale + zero_point).astype(np.uint8)  # de-scale
    test_img = np.array([test_img]).transpose(0, 2, 3, 1)
    interpreter.set_tensor(input['index'], test_img)
    start = datetime.datetime.now()
    interpreter.invoke()
    time = datetime.datetime.now() - start
    y = interpreter.get_tensor(output['index'])
    scale, zero_point = output['quantization']
    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
    y[..., 0] *= w  # x
    y[..., 1] *= h  # y
    y[..., 2] *= w  # w
    y[..., 3] *= h  # h
    pred = process_outs(y, conf_thres = conf)
    pred[:, :4] = scale_coords(test_img.shape[1:3], pred[:, :4], test_img0.shape).round()
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
    shape = test_img0.shape
    scores = pred[:,4]
    classes = pred[:,5].astype("uint8")
    if boxes is not None:
        draw(test_img0, boxes, scores, classes, data)
    return test_img0, time


def detect_video(video, interpreter, imgsz, data, conf):
    camera = cv2.VideoCapture(video)
    fps = camera.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    vout = cv2.VideoWriter()
    vout.open("OutPut"+video, fourcc, fps, sz, True)
    count = 0
    time_array = []
    while True:
        res, frame = camera.read()
        if count%2==0:
            vout.write(np.array(frame))
            count = count + 1
            continue

        if not res:
            break

        image, time = detect_image(Image.fromarray(frame), interpreter, imgsz, data, False, conf)
        print(image.shape)
        time_array.append(time)
        count += 1
        # cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()
    # cv2.destroyAllWindows()
    print(np.mean(time_array))


def run(weights=ROOT / 'yolov5s.pt', source=ROOT / 'data/images', imgsz=256, data="datasets/LPCV.yaml", conf=0.25):
    model_path, source, imgsz, data= opt.weights, opt.source, opt.imgsz, opt.data
    data = get_data_dict(data)['names']
    interpreter = tflite.Interpreter(model_path)
    interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
    # import tensorflow as tf

    # interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    if source.endswith("jpg") or source.endswith("jpeg"):
        image = Image.open(source)
        new_image, time = detect_image(image, interpreter, imgsz, data, source, conf)
        # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("image", new_image)
        # cv2.waitKey(0)
        cv2.imwrite(source[:len(source)-4]+"Output.jpg", new_image)
    elif source.endswith("m4v") or source.endswith("mp4"): 
        video = source
        detect_video(video, interpreter, imgsz, data, conf)
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
    parser.add_argument('--conf', type=float, default=0.25)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
