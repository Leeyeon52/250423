import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# 스쿼트 추적 관련 변수
squat_count = 0
is_squatting = False
knee_threshold_ratio = 0.65
hip_threshold_ratio = 0.75
initial_hip_y = None
thigh_length = None

# 얼굴 표정 관련 변수
expression = "Neutral"

def calculate_thigh_length(landmarks, image_height):
    # (이전 스쿼트 코드의 thigh_length 계산 함수)
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
    # (이전 스쿼트 코드의 무릎 임계값 확인 함수)
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
    # (이전 스쿼트 코드의 엉덩이 임계값 확인 함수)
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

def analyze_face_expression(landmarks, image_width, image_height):
    # (이전 얼굴 표정 분석 예시 함수 - 필요에 따라 더 정교하게 구현)
    if not landmarks:
        return "Neutral"

    landmarks = landmarks.landmark
    # 간단한 예시: 입의 세로 길이를 기준으로 웃음/찡그림 판단
    mouth_upper = np.array([landmarks[13].x * image_width, landmarks[13].y * image_height])
    mouth_lower = np.array([landmarks[14].x * image_width, landmarks[14].y * image_height])
    mouth_height = np.linalg.norm(mouth_upper - mouth_lower)

    # 간단한 예시: 눈썹 사이 거리로 찡그림 판단
    left_eyebrow_inner = np.array([landmarks[35].x * image_width, landmarks[35].y * image_height])
    right_eyebrow_inner = np.array([landmarks[295].x * image_width, landmarks[295].y * image_height])
    eyebrow_distance = np.linalg.norm(left_eyebrow_inner - right_eyebrow_inner)

    if mouth_height < 10 and eyebrow_distance > 30:
        return "Smiling"
    elif mouth_height > 20 and eyebrow_distance < 20:
        return "Frowning"
    else:
        return "Neutral"

# 웹캠 설정
cap = cv2.VideoCapture(0)
pTime = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose, \
    mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image)
        face_results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape
        pose_landmarks = pose_results.pose_landmarks
        face_landmarks = face_results.multi_face_landmarks

        # 스쿼트 감지 로직
        if pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if initial_hip_y is None and mp_pose.PoseLandmark.LEFT_HIP in pose_landmarks.landmark and mp_pose.PoseLandmark.RIGHT_HIP in pose_landmarks.landmark:
                initial_hip_y = (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                thigh_length = calculate_thigh_length(pose_landmarks.landmark, h)

            if initial_hip_y is not None and thigh_length is not None:
                knee_bent = is_knee_below_threshold(pose_landmarks.landmark, h, thigh_length)
                hip_low = is_hip_below_threshold(pose_landmarks.landmark, h, initial_hip_y)

                if knee_bent and hip_low and not is_squatting:
                    is_squatting = True
                elif is_squatting and not (knee_bent and hip_low):
                    squat_count += 1
                    is_squatting = False
                    print(f"스쿼트 횟수: {squat_count}")
                    initial_hip_y = None

            cv2.putText(image, f'Squat Count: {squat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            initial_hip_y = None
            is_squatting = False

        # 얼굴 표정 추적 로직
        if face_landmarks:
            for landmarks in face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

                expression = analyze_face_expression(landmarks, w, h)
                cv2.putText(image, f'Expression: {expression}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            expression = "No Face"

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Squat & Face Expression Tracker', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()