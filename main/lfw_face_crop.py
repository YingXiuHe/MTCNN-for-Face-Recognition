#-*-coding: utf-8-*-
#author: lxz-HXY
'''
代码是将lfw数据集中人脸图片截取下来并保存；
用于facenet人脸识别深度网络模型的训练；
人脸裁剪采用mtcnn算法；
'''
import os
import gc
import time
import logging
from cv2 import cv2
import tensorflow as tf 

# 加载mtcnn相关库和model
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector


def log_set():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(filename)s：%(lineno)d - %(levelname)s - %(message)s')
    
# 加载mtcnn模型
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


# 进行人脸裁剪，并将裁剪好的照片保存至新文件夹中，文件夹按照名字归类；
def crop_face(target_file):
    base_path = '/home/lxz/project/faceid/cropface/casia_crop_160'
    pic_list = list()
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    dirs = os.listdir(target_file)
    for dir in dirs:
        pics = os.listdir(target_file + '/' + dir)
        for pic in pics:
            fullpath = os.path.join(target_file + '/' + dir, pic)
            pic_list.append(fullpath)              
        face_dates = TestLoader(pic_list)
        allBoxes, _ = mtcnnDetector.detect_face(face_dates)
        for idx, imagepath in enumerate(pic_list):
            image = cv2.imread(imagepath)
            for box in allBoxes[idx]:
                x1 = int(box[0]) - 10
                y1 = int(box[1]) - 10
                x2 = int(box[2]) + 10
                y2 = int(box[3]) + 10
                # cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
                # cv2.imshow('asdas', image)
                # cv2.waitKey(0)    
                image = image[y1:y2, x1:x2]
                image = cv2.resize(image, (160, 160))
                save_path = base_path + '/' + str(dir)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                cv2.imwrite(os.path.join(save_path, str(pic.split('.')[0]) + '_' + str(idx) + '.jpg'), image)
                pic_list = list() # 清空列表，防止照片重复使用
                del x1, x2, y1, y2
                gc.collect()


'''
2019-9-12号记录：
代码存在问题，某些图片扣取下来之后，保存的图片为空！！！！！！
'''
if __name__ == '__main__':
    start = time.time()
    log_set()
    detectors = net('onet')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 40, threshold=[0.9, 0.6, 0.7])
    crop_face(target_file='/home/lxz/project/faceid/cropface/CASIA-FaceV5')
    end = time.time()
    logging.info('Finished operate!! Costed time {}/s'.format(int(end - start)))