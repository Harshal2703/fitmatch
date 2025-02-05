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
        
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        hip_angle = calculate_angle(shoulder, hip, ankle)

        if 170 <= hip_angle <= 190:  # More accurate range for a plank
            if not in_plank:
                start_time = time.time()
                in_plank = True
            elapsed_time = time.time() - start_time
            plank_time += elapsed_time
            feedback = "Good plank! Hold it steady."
        else:
            if in_plank:
                plank_time += time.time() - start_time
                in_plank = False
            feedback = "Keep your hips aligned and straight."

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        cv2.rectangle(image, (0, 0), (640, 60), (245, 117, 16), -1)
        cv2.putText(image, f'Plank Time: {int(plank_time)}s', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, feedback, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error in plank analysis: {e}")

    return image, plank_time, start_time, in_plank
