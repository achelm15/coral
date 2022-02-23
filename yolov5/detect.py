# import tflite_runtime.interpreter as tflite
import importlib
from importlib import util
from PIL import Image
import datetime
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
from sys import platform
from general import process_outs, get_data_dict, scale_coords, letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def draw(image, boxes, scores, classes, all_classes):
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box

        top = max(0, torch.floor(x1 + 0.5).int()).item()
        left = max(0, torch.floor(y1 + 0.5).int()).item()
        right = max(0, torch.floor(x2 + 0.5).int()).item()
        bottom = max(0, torch.floor(y2 + 0.5).int()).item()
        # print(top.item(), left, right, bottom) 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[int(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print(' box coordinate x,y,w,h: {0} '.format(box) + 'class: {0}, score: {1:.2f}'.format(all_classes[int(cl)], score) )


def detect_image(image, interpreter, imgsz, data, pathname, conf):
    
    if pathname:
        time4 = datetime.datetime.now()
        test_img0 = cv2.imread(pathname)
        print("SETUP ONLY LETTERBOX: ",datetime.datetime.now()-time4)
    else: 
        time4 = datetime.datetime.now()
        test_img0 = np.asarray(image)
        print("SETUP ONLY READ IMAGE: ",datetime.datetime.now()-time4)
    
    test_img = np.ascontiguousarray(letterbox(test_img0, imgsz, stride=64, auto=False)[0].transpose((2, 0, 1))[::-1])
    
    _, h, w = test_img.shape
    test_img = test_img.astype(np.float32)[None]
    test_img /= 255
    # device = torch.device('cpu')
    # test_img = torch.from_numpy(test_img).to(device)
    # test_img = test_img.float()  # uint8 to fp16/32
    # test_img /= 255  # 0 - 255 to 0.0 - 1.0
    # if len(test_img.shape) == 3:
    #     test_img = test_img[None]  # expand for batch dim
    # b, ch, h, w = test_img.shape  # batch, channel, height, width
    # input, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero_point = io[0]['quantization']
    print(scale,zero_point)
    test_img = (test_img / scale + zero_point).astype(np.uint8)  # de-scale
    test_img = torch.from_numpy(test_img).to('cpu').permute(0,2,3,1).cpu().numpy()
    interpreter.set_tensor(io[0]['index'], test_img)
    


    start = datetime.datetime.now()
    interpreter.invoke()
    time = datetime.datetime.now() - start


    y = interpreter.get_tensor(io[1]['index'])
    scale, zero_point = io[1]['quantization']
    print(scale,zero_point)
    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
    y[..., :4] *= [w, h, w, h]
    y = torch.tensor(y) if isinstance(y, np.ndarray) else y
    time2 = datetime.datetime.now()
    pred = process_outs(y, conf_thres = conf)[0]
    pred[:, :4] = scale_coords(test_img.shape[1:3], pred[:, :4], test_img0.shape).round()
    results = np.unique(pred[:,5], return_counts=True)
    results = ([(data[int(i)]+"s") for i in results[0]], results[1])
    result_s = "Found: "
    for x in range(0,len(results[0])):
        if x != len(results[0])-1:
            result_s+=str(int(results[1][x])) + " " + results[0][x] + ", "
        else:
            result_s+=str(int(results[1][x])) + " " + results[0][x]
    print("PROCESSING ", datetime.datetime.now()-time2)


    print(result_s)
    print(time)
    time3 = datetime.datetime.now()
    if pred[:,4] is not None:
        draw(test_img0, pred[:,:4], pred[:,4], pred[:,5].int(), data)
    print("DRAWING: ",datetime.datetime.now()-time3)

    return test_img0, time


def detect_video(video, interpreter, imgsz, data, conf):
    print(io,"afsdfa;sjdlfkjas;dlkfja;sdlkfjas;dlkfja;sldkfja;sldkjf")
    camera = cv2.VideoCapture(video)
    show = platform == "darwin" or platform =="win32"
    fps = camera.get(cv2.CAP_PROP_FPS)
    if show:
        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vout = cv2.VideoWriter()
    out = video.split('/')
    if len(out)==2:
        out = out[1][:len(out[1])-4]+".mp4"
    else:
        out = out[0][:len(out[0])-4]+".mp4"
    vout.open("OutPut"+out, fourcc, fps, sz, True)
    count = 0
    time_array = []
    total_det = []
    total_write = []
    frame_rate = []
    while True and count< 20:
        print(count)
        res, frame = camera.read()
        frame_start = datetime.datetime.now()
        time1 = datetime.datetime.now()
        image, time = detect_image(Image.fromarray(frame), interpreter, imgsz, data, False, conf)
        end1 = datetime.datetime.now()-time1
        print("TOTAL DETECTION: ", end1)
        total_det.append(end1)
        time_array.append(time)
        count += 1
        if show:
            cv2.imshow("detection", image)

        # Save the video frame by frame
        time1 = datetime.datetime.now()
        # vout.write(image)
        end1 = datetime.datetime.now()-time1
        print("IMAGE WRITE TIME: ", end1)
        total_write.append(end1)
        frame_end = datetime.datetime.now()-frame_start
        frame_rate.append(frame_end)
        if not res:
            break
        

        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()
    if show:
        cv2.destroyAllWindows()
    print(np.mean(time_array))
    print(np.mean(total_det))
    print(np.mean(total_write))
    print(np.mean(frame_rate))


def run(weights=ROOT / 'yolov5s.pt', source=ROOT / 'data/images', imgsz=256, data="datasets/LPCV.yaml", conf=0.25):
    model_path, source, imgsz, data= opt.weights, opt.source, opt.imgsz, opt.data
    data = get_data_dict(data)['names']
    spam_loader = util.find_spec('tflite_runtime')
    found = spam_loader is not None
    if found:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path)
        interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
    else:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    global io
    io = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if source.endswith("jpg") or source.endswith("jpeg"):
        image = Image.open(source)
        new_image, time = detect_image(image, interpreter, imgsz, data, source, conf)
        if platform == "darwin" or platform =="win32":
            cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("image", new_image)
            cv2.waitKey(0)
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
