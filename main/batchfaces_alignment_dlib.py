#-*-coding: utf-8-*-
#author: lxz-HXY

'''
采用68个关键点进行批量图片中人脸对齐操作
'''
import os
import sys
import dlib
import logging 
import time
import numpy as np
from cv2 import cv2 as cv2
 

def logset():
    #设置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(filename)s：%(lineno)d - %(levelname)s - %(message)s')
    

def load_module():
    '''
    1. 加载需要对齐的人脸照片；
    2. 加载特征点检测模型
    '''
    current_path = os.getcwd()
    logging.debug(current_path + '\n')
    model_path = current_path + '/main/dlib_model/shape_predictor_68_face_landmarks.dat'
    
    # cnn face_detection model
    # detector = dlib.cnn_face_detection_model_v1('/home/lxz/project/faceid/test_code/mmod_human_face_detector.dat')
    
    detector = dlib.get_frontal_face_detector()
    landmark = dlib.shape_predictor(model_path)
    logging.info('Landmark model load sucessed !!!')
    return detector, landmark


def face_aligniment(file_path, save_path):
    '''
    对文件夹中的照片进行批量的人脸对齐操作
    '''
    logging.info('Begin to alignment faces !!!')
    imgs = os.listdir(file_path)
    # print(imgs)
    for img in imgs:
        img_full_path = file_path + '/' + img
        bgr_imgs = cv2.imread(img_full_path)
        if bgr_imgs is None:
            logging.warn('Load pics failed !!! Please check file path !!!')
            exit()
        
        # 照片颜色通道转换：dlib检测的图片空间是RGB， cv2的颜色空间是BGR
        rgb_imgs = cv2.cvtColor(bgr_imgs, cv2.COLOR_BGR2RGB)
        face_location = detector(rgb_imgs, 1)
        if len(face_location) == 0:
            logging.warn('No face detected in pic: {}'.format(img_full_path))
            continue
        
        # 人脸关键点检测
        face_keypoints = dlib.full_object_detections()
        for location in face_location:
            face_keypoints.append(landmark(rgb_imgs, location))
            
        # 人脸对齐
        alignmented_face = dlib.get_face_chips(rgb_imgs, face_keypoints, size=120)
        logging.info('Alignment face sucessed: {}'.format(img_full_path))
        
        # 保存对齐后的人脸照片
        for image in alignmented_face:
            rgb_img = np.array(image).astype(np.uint8) # np数组
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow(img, bgr_img)
            # cv2.waitKey(100)
            cv2.imwrite(save_path + str(img.split('.')[0]) + '.jpg', bgr_img)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    start = time.time()
    logset()
    detector, landmark = load_module()
    save_path = 'aligmented/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    face_aligniment(file_path='/home/lxz/project/faceid/main/alignment', 
                                       save_path=save_path)
    end = time.time()
    logging.info('Operate Finished !!! Costed time:{} s'.format(end-start))    
