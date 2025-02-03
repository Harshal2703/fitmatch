import cv2
import mediapipe as mp
import numpy as np
import collections

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
pushup_counter = 0
stage = "Starting"
feedback = "Align your body properly"
angle_history = collections.deque(maxlen=5)  # Moving average filter

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)
            hip_angle = calculate_angle(shoulder, hip, knee)
            
            angle_history.append(elbow_angle)
            smooth_elbow_angle = np.mean(angle_history)
            
            # Ensure vertical movement of torso and prevent dumbbell detection
            if smooth_elbow_angle > 165 and abs(wrist[2] - shoulder[2]) < 0.1:
                stage = "Up"
                feedback = "Lower your body"
            elif smooth_elbow_angle < 80 and stage == "Up" and abs(wrist[2] - shoulder[2]) < 0.1:
                if shoulder_angle > 40:
                    feedback = "Keep elbows close!"
                elif hip_angle < 160:
                    feedback = "Keep body straight!"
                else:
                    pushup_counter += 1
                    feedback = "Good rep!"
                stage = "Down"
                
            cv2.rectangle(image, (0,0), (640, 60), (245,117,16), -1)
            cv2.putText(image, f'Reps: {pushup_counter} | Position: {stage}', (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, feedback, (10,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            
        except:
            pass
        
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        cv2.imshow('Pushup Analyzer', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
