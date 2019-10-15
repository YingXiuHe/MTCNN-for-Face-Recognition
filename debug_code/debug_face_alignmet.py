#-*-coding: utf-8-*-
# author: lxz-HXY

'''
使用68个人脸关键点进行人脸对齐操作;
人脸对齐操作采用dlib库
'''
import os 
import cv2
import dlib 
import logging
import numpy as np 
import face_recognition

# 加载mtcnn相关库和model
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector


def log():
    # 配置log格式
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')


def net(stage):
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = '/home/lxz/project/faceid/main/tmp/model/pnet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net,modelPath) 
    if stage in ['rnet', 'onet']:
        modelPath = '/home/lxz/project/faceid/main/tmp/model/rnet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = '/home/lxz/project/faceid/main/tmp/model/onet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath)
    return detectors


def alignment(pic_dir):
    landmarks_file = '/home/lxz/project/faceid/main/shape_predictor_68_face_landmarks.dat'
    landmarks = dlib.shape_predictor(landmarks_file)
    alig = list()
    pic_list = list()
    face_location = list()
    for pic in os.listdir('/home/lxz/project/faceid/main/alignment'):
        full_path = os.path.join('/home/lxz/project/faceid/main/alignment', pic)
    pic_list.append(full_path)
    
    bgr_img = cv2.imread(full_path)
    imgs = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    testDatas = TestLoader(pic_list)
    
    # 这里需要注意boxes坐标信息的处理
    allBoxes, _ = mtcnnDetector.detect_face(testDatas)
    for box in allBoxes[0]:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        face_location.append((y1, x2, y2, x1)) # 适当调节定位范围，防止人脸关键位置遗漏
        # alig.append((x1, y1))
        # alig.append((x2, y2))
        alig = [[(x1, y1),(x2, y2)]]
        
    logging.debug(face_location) # [(40, 198, 176, 95)]
    logging.debug(alig)  # [(95, 40), (198, 176)]
    
    faces = dlib.full_object_detection() # dlib.full_object_detection(arg0: dlib.rectangle, arg1: list)
    for face in alig:
        faces.append(landmarks(imgs, face))
    
    #进行人脸对齐
    images = dlib.get_face_chips(imgs, faces)
    
    cv_rgb_image = np.array(images).astype(np.uint8)# 先转换为numpy数组
    cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)# opencv下颜色空间为bgr，所以从rgb转换为bgr
    cv2.imshow('sssssss', cv_bgr_image)
    cv2.waitKey(100000000)
        
                


if __name__ == '__main__':
    log()
    detectors = net('onet')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    alignment('/home/lxz/project/faceid/main/alignment')
    