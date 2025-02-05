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

def calculate_accuracy(elbow_angle):
    # The ideal push-up angle at the bottom is around 90 degrees
    ideal_angle = 90
    accuracy = max(0, 100 - abs(ideal_angle - elbow_angle))
    return accuracy

def pushup_analysis(image, results, pushup_angle_history, pushup_counter, stage):
    feedback = ""
    try:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        pushup_angle_history.append(elbow_angle)
        smooth_elbow_angle = np.mean(pushup_angle_history[-5:])

        # Calculate accuracy
        accuracy = calculate_accuracy(smooth_elbow_angle)

        # Push-up detection logic
        if smooth_elbow_angle > 160:
            stage = "Up"
            feedback = "Lower your body"
        elif smooth_elbow_angle < 90 and stage == "Up":
            feedback = "Push up!"
            pushup_counter += 1
            stage = "Down"

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Display push-up feedback and accuracy on the image
        cv2.rectangle(image, (0, 0), (640, 80), (245, 117, 16), -1)  # Increased height to fit text
        cv2.putText(image, f'Reps: {pushup_counter} | Position: {stage}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Accuracy: {accuracy}%', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, feedback, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error in push-up analysis: {e}")

    return image, pushup_counter, stage, pushup_angle_history
