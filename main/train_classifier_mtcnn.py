#coding: utf-8 
#author: lxz-hxy

'''
人脸定位采用mtcnn
训练人脸特征模型：KNN
用到了face_recognition模块： pip安装即可
'''

import math
from sklearn import neighbors 
import os
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import logging 

# 加载mtcnn相关库和model
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


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            # 初始化
            pic_list = []
            face_location = []
            image = face_recognition.load_image_file(img_path)
            #print(img_path)
            pic_list.append(img_path)
            testDatas = TestLoader(pic_list)   
            # 这里需要注意boxes坐标信息的处理(与原始的mtcnn输出的坐标信息有区别)
            allBoxes, _ = mtcnnDetector.detect_face(testDatas)
            for box in allBoxes[0]:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
            face_location.append((y1-10, x2+12, y2+10, x1-12))
            logging.debug(face_location)
            
            if len(face_location) != 1:
                # 要是一张图中人脸数量大于一个，跳过这张图
                if verbose:
                    logging.warn("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_location) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_location, num_jitters=6)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            logging.info("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            #data = {"encoding": X, "name":y}
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    logset()
    start = time.time()
    logging.info("Start Training classifier...")
    detectors = net('onet')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 40, threshold=[0.9, 0.6, 0.7])
    classifier = train("/home/lxz/project/faceid/main/LFW", model_save_path="LFWtest_classifier_model.clf", n_neighbors=2)
    end = time.time()
    logging.info("Training complete! Time cost: {} s".format(int(end-start)))