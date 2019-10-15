# Author: 何秀颖 
# e-mail: yingxh1995@aliyun.com

MTCNN人脸检测 + face_recognition + KNN人脸识别 
(face_recognition模块的人脸定位采用dlib模块，本project采用mtcnn对其进行替换)

环境配置：
nvidia-smi:410.93
python=3.5
cuda=10.0
cudnn=v7.4
tf-gpu=1.13.1
face_recognition
opencv-python
dlib(需要源码编译，启用GPU加速)
scikit-learn
pickle

实现流程
1.将底库照片分好文件夹，训练KNN人脸特征分类器；
  对应代码： train_classifier.py [训练时需要注意代码内相关的路径,人脸定位采用dlib]
       或   train_classifier_mtcnn.py [人脸定位采用mtcnn，效果优于dlib]

  训练用的代码有两份： 一份代码训练用于图片人脸识别： test_code/train_classifier_mtcnn.py 生成的特征库文件为：***.clf文件
		      	    一份代码训练用于视频流人脸识别： test_code/encodings_faces_mtcnn.py 生成的特征库文件为： ***.pickle文件
  							
  训练数据集格式：
	  person-1(name)
		--1.jpg
		--2.jpg
		--**.jpg
		--****
	  person-2(name)
		--1.jpg
		--2.jpg
		--**.jpg
		--****
	  .......
	  persong-n(name)
		--1.jpg
		--2.jpg
		--**.jpg
		--****

2.测试识别 (这里需要注意几个参数的设定，能一定程度提升人脸识别的精度： model='large'; num_jitters(人脸re_sample的次数))
  1.单张照片人脸识别： 
	代码： recognize_faces_pic.py [初始化模型加载比较耗时间 40s左右，加载后识别速度稳定]
	      test_code/mtcnn_recognize_faces_pic.py [调用mtcnn检测模型，人脸识别效果更优]
  2.实时视频流人脸识别：
	代码： test_code/mtcnn_recognize_video.py 



# 2019-8-27 增加了人脸对齐操作的代码，一定程度上尝试提升人脸识别的精确度
代码为： /test_code/batchfaces_alignment_dlib.py  

