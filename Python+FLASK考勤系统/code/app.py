from flask import Flask, jsonify, request, render_template
import cv2
import dlib
from threading import Lock, Event
import os
import tempfile
import sys
import os
import uuid
import random
from flask import session
import base64


sys.path.append(
    os.path.abspath(
        "E:\Desktop\my\method"
    )
)

from method.detect import main_solve


app = Flask( __name__)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "E:\\Desktop\\my\\shape_predictor_68_face_landmarks.dat"
)

recognized_data = {}
data_lock = Lock()
video_capture = None
capture_thread = None
liveness_sessions = {}
IMAGE_DIRECTORY = "E:\Desktop\my\picture"


@app.route("/generate_question")
def generate_question():
    questions = [
        "blink eyes",
        "open mouth",
        "turn face right",
        "turn face left",
    ]
    index_question = random.randint(0, 3)
    question = questions[index_question]
    print("generate questioning")
    return jsonify({"question": question})


@app.route("/process_video", methods=["POST"])
def process_video():
    question = request.form["question"]
    video_file = request.files["video"]
    method = request.form["method"]

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    recognized_faces = []

    result_message = ""
    
    #引入我们的活体检测和人脸识别比对类
    my_solve = main_solve()

    if method == "local detect":
        # 活体检测处理整个视频流
        liveness_result = my_solve.detect_face_local(cap, question)
        if liveness_result:
            result_message = "Local Liveness detect confirmed"
    elif method == "network detect":
        print("network detect")
        
        # 综合处理方法(加上了API)
        liveness_result = my_solve.detect_face(cap, question)
        
        if liveness_result:
            result_message = "network Liveness detect completed"

    if liveness_result:
        print("Liveness confirmed")
        # 进行人脸识别
        cap.release()  # 首先释放当前的VideoCapture对象
        cap = cv2.VideoCapture(temp_video_path)  # 重新打开视频文件

        names = my_solve.recognize_instant_face(cap)
        
        if names:
            recognized_faces.append(names[0])
            print(f"Recognized faces: {names}")
        else:
            print("No faces recognized in the frame.")
    else:
        print("No valid frame available for recognition.")

    cap.release()
    os.remove(temp_video_path)
    os.rmdir(temp_dir)
    
    # 找到识别到的面孔对应的图像路径
    recognized_faces_images = []
    for face in recognized_faces:
        image_path = get_image_path(face)
        if image_path:
            encoded_image = encode_image_to_base64(image_path)
            recognized_faces_images.append(encoded_image)
            
    return jsonify(
        {
            "status": result_message,
            "data": {
                "faces": recognized_faces,
                "faces_photo": recognized_faces_images,
                "live_detected": liveness_result,
            },
        }
    )

def encode_image_to_base64(image_path):
    """
    将图像文件转换为base64编码
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_image_path(name):
    """
    根据名字查找图像文件路径
    """
    for filename in os.listdir(IMAGE_DIRECTORY):
        if name in filename:
            return os.path.join(IMAGE_DIRECTORY, filename)
    return None

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/results")
def results():
    with data_lock:
        print(f"Results: {recognized_data}")
        return jsonify(recognized_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
