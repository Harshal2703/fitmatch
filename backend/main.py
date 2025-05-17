from flask import Flask, request, render_template
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from thresholds.squat import squat_analysis
from thresholds.plank import plank_analysis
from thresholds.pushup import pushup_analysis
from thresholds.high_knees import high_knees_analysis
import time

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)  # [CHANGED] Added config to make pose more optimized for video streams

# Global variables for squat
squat_angle_history = []
squat_counter = 0
squat_stage = None

# Global variables for plank
plank_time = 0
start_time = None
in_plank = False

# Global variables for pushup
pushup_angle_history = []
pushup_counter = 0
pushup_stage = None

# Global variables for high knees
high_knees_history = []
high_knees_counter = 0
high_knees_stage = {'left_lifted': False, 'right_lifted': False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    global squat_angle_history, squat_counter, squat_stage
    global plank_time, start_time, in_plank
    global pushup_angle_history, pushup_counter, pushup_stage
    global high_knees_history, high_knees_counter, high_knees_stage

    data = request.get_json()
    
    # [CHANGED] Added error handling for missing data
    if not data or 'frame' not in data or 'exercise' not in data:
        return {'error': 'Invalid request payload'}, 400

    frame_data = data['frame'].split(',')[1]
    exercise = data['exercise']
    frame_bytes = base64.b64decode(frame_data)

    # Convert frame data to an image
    try:
        image = np.array(Image.open(BytesIO(frame_bytes)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Image decoding error: {e}")  # [CHANGED] Added error message
        return {'error': 'Could not decode image'}, 500

    # Process the image with MediaPipe Pose
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # [CHANGED] Added check for results and pose landmarks
    if not results or not results.pose_landmarks:
        return {'error': 'No pose landmarks detected'}, 204

    # Process based on exercise type
    if exercise == "squat":
        image, squat_counter, squat_stage, squat_angle_history = squat_analysis(
            image, results, squat_angle_history, squat_counter, squat_stage
        )
    elif exercise == "pushup":
        image, pushup_counter, pushup_stage, pushup_angle_history = pushup_analysis(
            image, results, pushup_angle_history, pushup_counter, pushup_stage
        )
    elif exercise == "plank":
        image, plank_time, start_time, in_plank = plank_analysis(
            image, results, plank_time, start_time, in_plank
        )
    elif exercise == "high_knees":
        image, high_knees_counter, high_knees_stage, high_knees_history = high_knees_analysis(
            image, results, high_knees_history, high_knees_counter, high_knees_stage
        )
    else:
        return {'error': 'Invalid exercise type'}, 400  # [CHANGED] Added fallback for invalid exercise name

    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == "__main__":
    app.run(debug=True)
