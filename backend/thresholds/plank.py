import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def plank_analysis(image, results, plank_time, start_time, in_plank):
    feedback = ""
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get key points for plank analysis
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angles for plank position
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)

        # Check if the user is in the plank position
        if 160 < hip_angle < 200 and 160 < knee_angle < 200:  # Ideal plank angles
            if not in_plank:
                start_time = time.time()  # Start the timer
                in_plank = True
            feedback = "Good plank position!"
        else:
            if in_plank:
                plank_time += time.time() - start_time  # Add elapsed time to total plank time
                in_plank = False
            feedback = "Adjust your position to plank"

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Display plank feedback and time on the image
        cv2.rectangle(image, (0, 0), (640, 60), (245, 117, 16), -1)
        cv2.putText(image, f'Plank Time: {int(plank_time)}s', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, feedback, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error in plank analysis: {e}")

    return image, plank_time, start_time, in_plank