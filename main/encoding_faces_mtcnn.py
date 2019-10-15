#-*-coding: utf-8-*-
#Author: lxz-hxy

'''
脚本执行指令： python encode_faces.py --dataset datasets --encodings encodings.pickle
生成的文件名为 ×××.pickle 文件
'''

# import the necessary packages
from imutils import paths
import logging
import face_recognition
import argparse
import pickle
import cv2
import os

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


def train():
    logging.debug("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images('/home/lxz/project/faceid/aligmented'))

    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        pic_list = []
        face_location = []
        # extract the person name from the image path
        logging.info("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        #rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pic_list.append(imagePath)
        testDatas = TestLoader(pic_list)
        # 这里需要注意boxes坐标信息的处理
        allBoxes, _ = mtcnnDetector.detect_face(testDatas)
        for box in allBoxes[0]:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
        face_location.append((y1-10, x2+12, y2+10, x1-12))

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(image, face_location, num_jitters=6)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    logging.info("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open('alignment.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()
    
if __name__=='__main__':
    logset()
    detectors = net('onet')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 40, threshold=[0.9, 0.6, 0.7]) 
    train()
