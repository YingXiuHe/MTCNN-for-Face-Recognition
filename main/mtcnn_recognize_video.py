# -*-coding: utf-8-*-
# author: lxz-hxy

'''
实时视频流中的人脸识别
mtcnn+face_recognition
'''
import tensorflow as tf
import face_recognition
import numpy as np
import logging 
import pickle
import time
import cv2
import os 

# 载入mtcnn相关模块
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector


def logset():
    #设置日志格式
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


def recognize():
    logging.info("[INFO]:Starting video stream...")
    
    # rtsp = 'rtsp://admin:***************'
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        boxes = []
        _, frame = cap.read()
        image = np.array(frame)
        allBoxes, _ = mtcnnDetector.detect_video(image)
        for box in allBoxes:
            x_1 = int(box[0])
            y_1 = int(box[1])
            x_2 = int(box[2])
            y_2 = int(box[3])
            boxes.append((y_1-10, x_2+12, y_2+10, x_1-12))
        
        start = time.time()
        #num_jitters（re-sample人脸的次数） 参数的设定，数值越大精度相对会高，但是速度会慢；
        encodings = face_recognition.face_encodings(frame, boxes, num_jitters=3)
        end = time.time()
        if len(boxes) != 0:
            logging.info('[INFO]:Encoding face costed: {} s'.format(end-start))
        names = []
        
        for encoding in encodings:
            # distance between faces to consider it a match, optimize is 0.6
            matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.45)            
            name = 'Unknown'
            if True in matches:
                matchesidx = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchesidx:
                    name = data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key = counts.get)
                logging.debug(name)
            names.append(name)             
        
        #绘制检测框 + 人脸识别结果
        for ((top, right, bottom, left), name) in zip(boxes, names):
            y1 = int(top)
            x1 = int(right)
            y2 = int(bottom)
            x2 = int(left)
            cv2.rectangle(frame, (x2, y1), (x1, y2), (0, 0, 255), 2)
            cv2.putText(frame, name, (x2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)        
        cv2.imshow('Recognize', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    logset()
    logging.info(tf.__version__)     
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    detectors = net('onet')
    mtcnnDetector = MtcnnDetector(detectors = detectors, min_face_size = 40, threshold = [0.9, 0.6, 0.7])
    data = pickle.loads(open('/home/lxz/project/faceid/mtcnn_video_classifier.pickle', "rb").read()) 
    logging.info("[INFO]:Load Encodings Database Sucessed!!")
    recognize() #开始识别    