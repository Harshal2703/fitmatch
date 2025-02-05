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
import time

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Global variables for squat
squat_angle_history = []
squat_counter = 0
squat_stage = None

# Global variables for plank
plank_time = 0  # Total time spent in plank position
start_time = None  # Timestamp when plank position is entered
in_plank = False  # Boolean to track if user is in plank position

# Global variables for pushup
pushup_angle_history = []
pushup_counter = 0
pushup_stage = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    global squat_angle_history, squat_counter, squat_stage
    global plank_time, start_time, in_plank
    global pushup_angle_history, pushup_counter, pushup_stage

    # Get frame data and exercise type from the request
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    exercise = data['exercise']
    frame_bytes = base64.b64decode(frame_data)

    # Convert frame data to an image
    image = np.array(Image.open(BytesIO(frame_bytes)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process the image with MediaPipe Pose
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_rgb is not None:
        results = pose.process(image_rgb)
    else:
        print("Image is empty or not valid")

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

    # Encode the processed image to JPEG and return it
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}


if __name__ == "__main__":
    app.run(debug=True)