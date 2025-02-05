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

def calculate_knee_accuracy(knee_height, hip_height):
    ideal_ratio = 0.8  # Adjusted for higher knee lift
    ratio = knee_height / hip_height if hip_height != 0 else 0
    accuracy = min(100, max(0, (1 - ratio) / (1 - ideal_ratio) * 100))  # Inverted ratio calculation
    return accuracy

def high_knees_analysis(image, results, angle_history, knee_lift_counter, stage):
    feedback = ""
    left_lifted = stage.get('left_lifted', False)
    right_lifted = stage.get('right_lifted', False)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Extract landmarks for left leg
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Extract landmarks for right leg
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Smooth angles using a window of 3 frames
        angle_history.append((left_knee_angle, right_knee_angle))
        window_size = 3
        smoothed_left = np.mean([a[0] for a in angle_history[-window_size:]])
        smoothed_right = np.mean([a[1] for a in angle_history[-window_size:]])

        # Threshold for knee lift detection
        angle_threshold = 80

        # Check left knee lift
        if smoothed_left < angle_threshold:
            if not left_lifted:
                knee_lift_counter += 1
                left_lifted = True
                feedback = "Left knee up!"
        else:
            left_lifted = False

        # Check right knee lift
        if smoothed_right < angle_threshold:
            if not right_lifted:
                knee_lift_counter += 1
                right_lifted = True
                feedback = "Right knee up!"
        else:
            right_lifted = False

        # Update stage with current lifted states
        stage = {
            'left_lifted': left_lifted,
            'right_lifted': right_lifted
        }

        # Calculate knee lift accuracy (using y-coordinates)
        left_knee_y = left_knee[1]
        right_knee_y = right_knee[1]
        hip_y = min(left_hip[1], right_hip[1])
        
        # Correct accuracy calculation (lower y means higher position)
        left_accuracy = calculate_knee_accuracy(left_knee_y, hip_y)
        right_accuracy = calculate_knee_accuracy(right_knee_y, hip_y)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Display information
        cv2.rectangle(image, (0, 0), (640, 80), (245, 117, 16), -1)
        cv2.putText(image, f'Count: {knee_lift_counter}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Left Acc: {left_accuracy:.1f}%', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Acc: {right_accuracy:.1f}%', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, feedback, (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error in high knees analysis: {e}")
        stage = {'left_lifted': False, 'right_lifted': False}  # Reset on error

    return image, knee_lift_counter, stage, angle_history