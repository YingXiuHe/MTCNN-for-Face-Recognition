#coding: utf-8
#author: lxz-hxy

'''
人脸定位：mtcnn
测试模型在照片上的人脸识别情况
'''

import os 
from cv2 import cv2 as cv2
import time
import pickle
import logging
import numpy as np 
import face_recognition
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 加载mtcnn相关库和model
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
        detectors[0] = FcnDetector(P_Net,modelPath) 
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

# distance_threshold数值的设定越小比对则越严格， 0.6为默认值
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.45):       
    pic_list = []
    face_location = []
    
    #加载分类模型
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
   
    pic_list.append(X_img_path)
    testDatas = TestLoader(pic_list)
    
    # 这里需要注意boxes坐标信息的处理
    allBoxes, _ = mtcnnDetector.detect_face(testDatas)
    for box in allBoxes[0]:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        face_location.append((y1-10, x2+12, y2+10, x1-12)) # 适当调节定位范围，防止人脸关键位置遗漏
    logging.debug(face_location)
   
    # If no faces are found in the image, return an empty result.
    if len(allBoxes[0]) == 0:
        logging.warn('No face find in pic!!!!')
        return []

    # Find encodings for faces in the test iamge
    X_img = face_recognition.load_image_file(X_img_path)

    star = time.time()
    # face_encodings api里面有个参数设定 num_jitters 人脸采样的次数，次数越多，精度相对会高，默认为1 
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=face_location, num_jitters=3)
    end = time.time()
    logging.info('face_encodings cost: {} s'.format(end-star))

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_location))]
    
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_location, are_matches)]


def show_results(img_path, predictions):
    image = cv2.imread(img_path)
    for name, (top, right, bottom, left) in predictions:
        cv2.putText(image, str(name), (left, top-10), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0,255,0))
        cv2.rectangle(image, (left, top),(right, bottom), (0, 0, 255), 2)
        cv2.imwrite(str(img_path)+'.jpg', image)   
    cv2.imshow('Recognition_Results', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

  
if __name__=='__main__':
    logset()
    detectors = net('onet')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 40, threshold=[0.9, 0.6, 0.7]) 
    for image_file in os.listdir("/home/lxz/project/faceid/datasets/test"):
        full_file_path = os.path.join("/home/lxz/project/faceid/datasets/test", image_file)
        
        predictions = predict(full_file_path, model_path="/home/lxz/project/faceid/LFW_classifier_model.clf")
        
        show_results(os.path.join("/home/lxz/project/faceid/datasets/test", image_file), predictions)
