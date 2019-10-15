#coding: utf-8
#author: lxz-hxy

'''
测试模型在照片上的人脸识别情况
'''
import os 
import cv2
import time
import pickle
import face_recognition
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):    
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    
    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_results(img_path, predictions):
    image = cv2.imread(img_path)
    for name, (top, right, bottom, left) in predictions:
        cv2.putText(image, str(name), (left, top-23), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0,255,0))
        cv2.rectangle(image, (left, top-15),(right, bottom+15), (0, 0, 255), 2)
        
    cv2.imshow('Recognition_Results', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

'''
Problem：初始化模型的加载速度比较慢40s左右
'''   

if __name__=='__main__':
    for image_file in os.listdir("/home/lxz/project/faceid/datasets/test"):
        full_file_path = os.path.join("/home/lxz/project/faceid/datasets/test", image_file)
        predictions = predict(full_file_path, model_path="/home/lxz/project/faceid/model/classifier_model.clf")
        show_results(os.path.join("/home/lxz/project/faceid/datasets/test", image_file), predictions)
