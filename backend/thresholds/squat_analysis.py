import cv2
import numpy as np
import mediapipe as mp

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

def squat_analysis(image, results, angle_history, squat_counter, stage):
    feedback = ""
    try:
        landmarks = results.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate hip angle
        hip_angle = calculate_angle(hip, knee, ankle)
        angle_history.append(hip_angle)
        smooth_hip_angle = np.mean(angle_history[-5:])

        # Squat detection logic
        if smooth_hip_angle > 160:
            stage = "Up"
            feedback = "Lower your body"
        elif smooth_hip_angle < 90 and stage == "Up":
            feedback = "Drive up!"
            squat_counter += 1
            stage = "Down"

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Display squat feedback on the image
        cv2.rectangle(image, (0, 0), (640, 60), (245, 117, 16), -1)
        cv2.putText(image, f'Reps: {squat_counter} | Position: {stage}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, feedback, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error in squat analysis: {e}")

    return image, squat_counter, stage, angle_history
