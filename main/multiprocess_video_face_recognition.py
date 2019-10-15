#-*-coding: utf-8-*-
#author: lxz-hxy
#e-mail: yingxh1995@aliyun.com
'''
采用多进程进行实时人脸识别
mtcnn人脸定位+ face-recognition
'''
import os
import gc  
import time
import pickle 
import logging
import numpy as np
from cv2 import cv2 as cv2
import face_recognition
import tensorflow as tf 
from multiprocessing import Process, Manager
import multiprocessing as mp 


# 载入mtcnn相关模块
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector


def logset():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(filename)s：%(lineno)d - %(levelname)s - %(message)s')


def net(stage):
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = '/home/lxz/project/faceid/main/tmp/model/pnet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        logging.info("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net, modelPath) 
    if stage in ['rnet', 'onet']:
        modelPath = '/home/lxz/project/faceid/main/tmp/model/rnet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        logging.info("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = '/home/lxz/project/faceid/main/tmp/model/onet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        logging.info("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath)
    return detectors


def receive(stack):
    logging.info('[INFO]Receive the video..........')
    
    top = 500
    # rtsp = 'rtsp://admin:*****************'
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        if ret:
            stack.append(frame)
            if len(stack) >= top:
                logging.info("stack full, begin to collect it......")
                del stack[:450]
                gc.collect()


def recognize(stack):
    logging.info("[INFO]:Starting video stream...")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config = config)
    data = pickle.loads(open('/home/lxz/project/faceid/alignment.pickle', "rb").read())
    detectors = net('onet')
    mtcnnDetector = MtcnnDetector(detectors = detectors, min_face_size = 60, threshold = [0.9, 0.6, 0.7])
    logging.info('MTCNN/KNN Model load sucessed !!!!')
    while True:
        if len(stack) > 20:
            boxes = []
            frame = stack.pop()
            image = np.array(frame)
            allBoxes, _ = mtcnnDetector.detect_video(image)
            for box in allBoxes:
                x_1 = int(box[0])
                y_1 = int(box[1])
                x_2 = int(box[2])
                y_2 = int(box[3])
                boxes.append((y_1-10, x_2+12, y_2+10, x_1-12))
            logging.debug(boxes)
            start = time.time()
            # num_jitters（re-sample人脸的次数）参数的设定，数值越大精度相对会高，但是速度会慢；
            encodings = face_recognition.face_encodings(frame, boxes, num_jitters=6)
            end = time.time()
            logging.info('[INFO]:Encoding face costed: {} s'.format(end-start))
            print('encode time is {}ms'.format((end-start)*1000))
            names = []
            
            for encoding in encodings:
                # distance between faces to consider it a match, optimize is 0.6
                matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.35)            
                name = 'Stranger'
                if True in matches:
                    matchesidx = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchesidx:
                        name = data['names'][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key = counts.get)
                    logging.debug(name)
                names.append(name)             
            
            # 绘制检测框 + 人脸识别结果
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # print(name)
                y1 = int(top)
                x1 = int(right)
                y2 = int(bottom)
                x2 = int(left)
                cv2.rectangle(frame, (x2, y1), (x1, y2), (0, 0, 255), 2)
                if name == 'Stranger':
                    cv2.putText(frame, name, (x2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    print(name)
                    cv2.putText(frame, name, (x2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Recognize-no-alignment', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logset()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    mp.set_start_method('spawn')
    p = Manager().list()
    p1 = Process(target=receive, args=(p,))
    p2 = Process(target=recognize, args=(p,))
    p1.start()
    p2.start()
    p1.join()
    p2.terminate()
