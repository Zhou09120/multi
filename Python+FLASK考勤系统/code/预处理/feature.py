import os
import dlib
import cv2
import numpy as np
import pickle

# 设置路径
image_folder = 'picture'  # 你的图片库路径
features_folder = 'face_feature'  # 你要存储特征的路径

# 创建文件夹
if not os.path.exists(features_folder):
    os.makedirs(features_folder)

# 加载 dlib 的人脸检测器和人脸特征提取器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def extract_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)
    return None

# 遍历图片库，提取人脸特征并存储
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        features = extract_features(image_path)
        if features is not None:
            feature_path = os.path.join(features_folder, os.path.splitext(filename)[0] + '.npy')
            np.save(feature_path, features)

print("人脸特征提取完成。")
