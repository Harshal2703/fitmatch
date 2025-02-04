from flask import Flask, request, render_template
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from thresholds.squat_analysis import squat_analysis

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

angle_history = []
squat_counter = 0
stage = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    global angle_history, squat_counter, stage
    
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    exercise = data['exercise'] 
    frame_bytes = base64.b64decode(frame_data)
    
    image = np.array(Image.open(BytesIO(frame_bytes)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_rgb is not None:
        results = pose.process(image_rgb)
    else:
        print("Image is empty or not valid")
    if exercise == "squat":
        image, squat_counter, stage, angle_history = squat_analysis(image, results, angle_history, squat_counter, stage)
    elif exercise == "pushup":
        pass
    elif exercise == "plank":
        pass

    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == "__main__":
    app.run(debug=True)
