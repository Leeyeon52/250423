import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 스쿼트 추적 관련 변수
squat_count = 0
is_squatting = False
knee_threshold_ratio = 0.65  # 허벅지 길이 대비 무릎 높이 비율 (조절 필요)
hip_threshold_ratio = 0.75   # 초기 엉덩이 높이 대비 현재 엉덩이 높이 비율 (조절 필요)
initial_hip_y = None
thigh_length = None

def calculate_thigh_length(landmarks, image_height):
    if landmarks and mp_pose.PoseLandmark.LEFT_HIP in landmarks and mp_pose.PoseLandmark.LEFT_KNEE in landmarks:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        return abs(left_hip.y - left_knee.y) * image_height
    elif landmarks and mp_pose.PoseLandmark.RIGHT_HIP in landmarks and mp_pose.PoseLandmark.RIGHT_KNEE in landmarks:
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        return abs(right_hip.y - right_knee.y) * image_height
    return None

def is_knee_below_threshold(landmarks, image_height, thigh_length, threshold_ratio=knee_threshold_ratio):
    if not landmarks or thigh_length is None:
        return False
    left_hip = landmarks.get(mp_pose.PoseLandmark.LEFT_HIP)
    left_knee = landmarks.get(mp_pose.PoseLandmark.LEFT_KNEE)
    right_hip = landmarks.get(mp_pose.PoseLandmark.RIGHT_HIP)
    right_knee = landmarks.get(mp_pose.PoseLandmark.RIGHT_KNEE)

    if left_hip and left_knee:
        return left_knee.y * image_height > left_hip.y * image_height + thigh_length * threshold_ratio
    elif right_hip and right_knee:
        return right_knee.y * image_height > right_hip.y * image_height + thigh_length * threshold_ratio
    return False

def is_hip_below_threshold(landmarks, image_height, initial_hip_y, threshold_ratio=hip_threshold_ratio):
    if not landmarks or initial_hip_y is None:
        return False
    left_hip = landmarks.get(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = landmarks.get(mp_pose.PoseLandmark.RIGHT_HIP)

    current_hip_y = None
    if left_hip and right_hip:
        current_hip_y = (left_hip.y + right_hip.y) / 2 * image_height
    elif left_hip:
        current_hip_y = left_hip.y * image_height
    elif right_hip:
        current_hip_y = right_hip.y * image_height

    if current_hip_y is not None:
        return current_hip_y > initial_hip_y * threshold_ratio
    return False

# 웹캠 설정 (실시간 추적)
cap = cv2.VideoCapture(0)
pTime = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        landmarks = {}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = landmark
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 초기 엉덩이 높이 설정 (첫 스쿼트 시도 시)
            if initial_hip_y is None and mp_pose.PoseLandmark.LEFT_HIP in landmarks and mp_pose.PoseLandmark.RIGHT_HIP in landmarks:
                initial_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                thigh_length = calculate_thigh_length(landmarks, h) # 초기 허벅지 길이도 설정

            if initial_hip_y is not None and thigh_length is not None:
                knee_bent = is_knee_below_threshold(landmarks, h, thigh_length)
                hip_low = is_hip_below_threshold(landmarks, h, initial_hip_y)

                if knee_bent and hip_low and not is_squatting:
                    is_squatting = True
                elif is_squatting and not (knee_bent and hip_low):
                    squat_count += 1
                    is_squatting = False
                    print(f"스쿼트 횟수: {squat_count}")
                    initial_hip_y = None # 스쿼트 완료 후 초기 엉덩이 높이 리셋

            cv2.putText(image, f'Squat Count: {squat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # 랜드마크가 감지되지 않으면 초기 엉덩이 높이 리셋
            initial_hip_y = None
            is_squatting = False

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Real-time Squat Tracker', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()