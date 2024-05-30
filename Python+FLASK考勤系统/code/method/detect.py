import cv2
import dlib
import face_recognition
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
from datetime import datetime  # 可以用于获取当前的时间

import os
import numpy as np

from util import BaiduApiUtil

TOLERANCE = 0.42
SET_SIZE = 0.33
PHOTO_FOLDER_PATH = "E://Desktop//my//_photo"

# 主界面
class main_solve():
    def __init__(self):
        self.detector = None  # 人脸检测器
        self.predictor = None  # 特征点检测器
        self.face_rec_model = None 
            
        # 闪烁阈值
        self.EAR_THRESH = None
        self.MOUTH_THRESH = None
        # 总闪烁次数
        self.eye_flash_counter = None
        self.mouth_open_counter = None
        self.turn_left_counter = None
        self.turn_right_counter = None
        # 连续帧数阈值
        self.EAR_CONSTANT_FRAMES = None
        self.MOUTH_CONSTANT_FRAMES = None
        self.LEFT_CONSTANT_FRAMES = None
        self.RIGHT_CONSTANT_FRAMES = None
        # 连续帧计数器
        self.eye_flash_continuous_frame = 0
        self.mouth_open_continuous_frame = 0
        self.turn_left_continuous_frame = 0
        self.turn_right_continuous_frame = 0

        #已知人脸数据集
        self.known_faces = self.features()
        
        # 百度API
        self.api = BaiduApiUtil
        
        #用于测试人脸的    
        self.output_dir = "E:\Desktop\my\cap_frame"
        self.frame_count = 0
        
        
    def features(self):
        # 加载预存储的人脸特征
        features_folder = 'face_feature'
        known_faces = {}
        for filename in os.listdir(features_folder):
            if filename.endswith('.npy'):
                person_id = os.path.splitext(filename)[0]
                features = np.load(os.path.join(features_folder, filename))
                known_faces[person_id] = features
        return known_faces


    # ----------------- 人脸识别 ------------------ #
    # 瞬时人脸识别
    def recognize_instant_face(self,cap):
        # 超过10帧判断识别失败
        total_frame = 0
        self.frame_count = 0
        # 当 检测成功 或 超过10帧 时退出循环
        while True:
            # 获取摄像头画面帧 frame
            ret, frame = cap.read()
            # 帧数加一
            total_frame += 1
            # 对帧进行裁剪处理，减少计算量
            #small_frame = cv2.resize(frame, (0, 0), fx=SET_SIZE, fy=SET_SIZE)

            #gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.detector is None:
                self.detector = dlib.get_frontal_face_detector()
            if self.predictor is None:
                self.predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')
            if self.face_rec_model is None:
                self.face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
                
            faces = self.detector(gray, 1)
            
            matched_id = []
            print(len(faces))
            
            if len(faces) == 0:
                break
        
            for face in faces:
                
                #被检测的图片的存储
                frame_filename = os.path.join(self.output_dir, f'frame_{self.frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)
                self.frame_count = 1 + self.frame_count
                
                
                shape = self.predictor(gray, face)
                face_descriptor = self.face_rec_model.compute_face_descriptor(frame, shape)
                face_features = np.array(face_descriptor)
                
                if(self.compare_faces(face_features) != None):
                    matched_id.append(self.compare_faces(face_features))

            if matched_id:
                return matched_id
            # 超时退出
            if total_frame >= 10.0:
                return matched_id

    def compare_faces(self,face_features):
        distances = {}
        for img_path, known_features in self.known_faces.items():
            distance = np.linalg.norm(known_features - face_features)
            distances[img_path] = distance
            print(img_path, distance)
        
        # Find the best match
        best_match = min(distances, key=distances.get)
        best_match_distance = distances[best_match]
        
        if best_match_distance < 0.6:    # 距离阈值可以根据需要调整
            return best_match
        
        return None
        
    # ----------------- 活体检测 ------------------ #
    # 整体活体检测
    def detect_face(self,cap,question):
        if self.api.network_connect_judge():
            if not self.detect_face_network(cap):
                return False
        if not self.detect_face_local(cap, question):
            return False
        return True

    # 联网活体检测
    def detect_face_network(self,cap):
        ret, frame = cap.read()
        frame_location = face_recognition.face_locations(frame)
        if len(frame_location) == 0:
            #未检测到人脸
            return False
        else:
            global PHOTO_FOLDER_PATH
            shot_path = PHOTO_FOLDER_PATH + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
            cv2.imwrite(shot_path, frame)

            # 百度API进行活体检测
            if not self.api.face_api_invoke(shot_path):
                #os.remove(shot_path)
                #'未通过活体检测'
                return False
            else:
                os.remove(shot_path)
                return True

    # 计算眼长宽比例 EAR值
    def count_EAR(self,eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        EAR = (A + B) / (2.0 * C)
        return EAR

    # 计算嘴长宽比例 MAR值
    def count_MAR(self,mouth):
        A = dist.euclidean(mouth[1], mouth[11])
        B = dist.euclidean(mouth[2], mouth[10])
        C = dist.euclidean(mouth[3], mouth[9])
        D = dist.euclidean(mouth[4], mouth[8])
        E = dist.euclidean(mouth[5], mouth[7])
        F = dist.euclidean(mouth[0], mouth[6])  # 水平欧几里德距离
        ratio = (A + B + C + D + E) / (5.0 * F)
        return ratio

    # 计算左右脸转动比例 FR值
    def count_FR(self,face):
        rightA = dist.euclidean(face[0], face[27])
        rightB = dist.euclidean(face[2], face[30])
        rightC = dist.euclidean(face[4], face[48])
        leftA = dist.euclidean(face[16], face[27])
        leftB = dist.euclidean(face[14], face[30])
        leftC = dist.euclidean(face[12], face[54])
        ratioA = rightA / leftA
        ratioB = rightB / leftB
        ratioC = rightC / leftC
        ratio = (ratioA + ratioB + ratioC) / 3
        return ratio

    
    # 本地活体检测
    def detect_face_local(self, cap, question):
        # 特征点检测器首次加载比较慢，通过判断减少后面加载的速度
        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()
        if self.predictor is None:
            self.predictor = dlib.shape_predictor('E:\\Desktop\\my\\shape_predictor_68_face_landmarks.dat')

        # 闪烁阈值
        self.EAR_THRESH = 0.25
        self.MOUTH_THRESH = 0.5

        # 总闪烁次数
        self.eye_flash_counter = 0
        self.mouth_open_counter = 0
        self.turn_left_counter = 0
        self.turn_right_counter = 0

        # 连续帧数阈值
        self.EAR_CONSTANT_FRAMES = 2
        self.MOUTH_CONSTANT_FRAMES = 2
        self.LEFT_CONSTANT_FRAMES = 3
        self.RIGHT_CONSTANT_FRAMES = 3

        # 连续帧计数器
        self.eye_flash_continuous_frame = 0
        self.mouth_open_continuous_frame = 0
        self.turn_left_continuous_frame = 0
        self.turn_right_continuous_frame = 0

        # 当前总帧数
        total_frame_counter = 0
        
        now_flag = 0

        # 抓取面部特征点的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        while total_frame_counter <= 100:
            ret, frame = cap.read()
            total_frame_counter += 1
            frame = imutils.resize(frame)       #方便处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)          #找到的人脸集合

            if len(rects) == 1:
                shape = self.predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)

                # 提取面部坐标
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]

                # 计算长宽比
                left_EAR = self.count_EAR(left_eye)
                right_EAR = self.count_EAR(right_eye)
                mouth_MAR = self.count_MAR(mouth)
                leftRight_FR = self.count_FR(shape)
                average_EAR = (left_EAR + right_EAR) / 2.0
                
                #print("left_ERA:",left_EAR)
                #print("right_ERA:",right_EAR)
                #print("mouth_MAR:",mouth_MAR)
                print("leftRight_FR:",leftRight_FR)
                #print("average_EAR:",average_EAR)
                

                # 计算左眼、右眼、嘴巴的凸包
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)

                # 可视化
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

                print(question)
                if question == 'turn face left':
                    if self.turn_left_counter > 0:
                        #now_flag += 1
                        #....
                        print("1")
                        return True
                    else:
                        self.check_left_turn(leftRight_FR)
                        self.turn_right_counter = 0
                        self.mouth_open_counter = 0
                        self.eye_flash_counter = 0
                        

                elif question == "turn face right":
                    if self.turn_right_counter > 0:
                        print("2")
                        return True
                    else:
                        self.check_right_turn(leftRight_FR)
                        self.turn_left_counter = 0
                        self.mouth_open_counter = 0
                        self.eye_flash_counter = 0

                elif question == "open mouth":
                    if self.mouth_open_counter > 0:
                        print("3")
                        return True

                    else:
                        self.check_mouth_open(mouth_MAR)
                        self.turn_right_counter = 0
                        self.turn_left_counter = 0
                        self.eye_flash_counter = 0

                elif question == "blink eyes":
                    if self.eye_flash_counter > 0:
                        print("4")
                        return True
                    else:
                        self.check_eye_flash(average_EAR)
                        self.turn_right_counter = 0
                        self.turn_left_counter = 0
                        self.mouth_open_counter = 0

            elif len(rects) == 0:
                continue

            elif len(rects) > 1:
                continue

        return False

    def check_eye_flash(self, average_EAR):
        #闭眼情况，使得self.eye_flash_continuous_frame不等于0
        if average_EAR < self.EAR_THRESH:
            self.eye_flash_continuous_frame += 1
        #睁眼情况，前面存在眨眼
        else:
            if self.eye_flash_continuous_frame >= self.EAR_CONSTANT_FRAMES:
                self.eye_flash_counter += 1
            self.eye_flash_continuous_frame = 0

    def check_mouth_open(self, mouth_MAR):
        if mouth_MAR > self.MOUTH_THRESH:
            self.mouth_open_continuous_frame += 1
        else:
            if self.mouth_open_continuous_frame >= self.MOUTH_CONSTANT_FRAMES:
                self.mouth_open_counter += 1
            self.mouth_open_continuous_frame = 0

    def check_right_turn(self, leftRight_FR):
        if leftRight_FR <= 0.5:
            self.turn_right_continuous_frame += 1
        else:
            if self.turn_right_continuous_frame >= self.RIGHT_CONSTANT_FRAMES:
                self.turn_right_counter += 1
            self.turn_right_continuous_frame = 0

    #检测左转帧数
    def check_left_turn(self, leftRight_FR):
        if leftRight_FR >= 2.0:
            print(self.turn_left_continuous_frame)
            self.turn_left_continuous_frame += 1
        else:
            if self.turn_left_continuous_frame >= self.LEFT_CONSTANT_FRAMES:
                self.turn_left_counter += 1
            self.turn_left_continuous_frame = 0

