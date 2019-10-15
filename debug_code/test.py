import face_recognition

image = face_recognition.load_image_file('/home/lxz/project/faceid/main/11.jpg')
landmark = face_recognition.face_landmarks(image)
print(landmark)
