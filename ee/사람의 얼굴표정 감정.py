import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 감정 분석 관련 변수
emotion = "Neutral"

def calculate_eyebrow_expression(landmarks):
    # 눈썹 관련 랜드마크 인덱스 (예시)
    left_eyebrow_upper = np.array([landmarks[22].x, landmarks[22].y])
    left_eyebrow_lower = np.array([landmarks[23].x, landmarks[23].y])
    right_eyebrow_upper = np.array([landmarks[263].x, landmarks[263].y])
    right_eyebrow_lower = np.array([landmarks[264].x, landmarks[264].y])

    # 눈썹 사이 거리 (찡그림)
    eyebrow_inner_dist = np.linalg.norm(left_eyebrow_lower - right_eyebrow_lower)

    # 눈썹-눈 거리 (놀람)
    left_eye_upper = np.array([landmarks[159].x, landmarks[159].y])
    left_eye_lower = np.array([landmarks[145].x, landmarks[145].y])
    right_eye_upper = np.array([landmarks[386].x, landmarks[386].y])
    right_eye_lower = np.array([landmarks[374].x, landmarks[374].y])
    left_eyebrow_to_eye = np.linalg.norm(left_eyebrow_lower - left_eye_upper)
    right_eyebrow_to_eye = np.linalg.norm(right_eyebrow_lower - right_eye_upper)

    if eyebrow_inner_dist < 0.03:  # 임계값 조절 필요
        return "Angry/Focused"
    elif left_eyebrow_to_eye > 0.07 and right_eyebrow_to_eye > 0.07: # 임계값 조절 필요
        return "Surprised"
    return None

def calculate_mouth_expression(landmarks):
    # 입 관련 랜드마크 인덱스 (예시)
    mouth_left = np.array([landmarks[308].x, landmarks[308].y])
    mouth_right = np.array([landmarks[78].x, landmarks[78].y])
    mouth_upper = np.array([landmarks[13].x, landmarks[13].y])
    mouth_lower = np.array([landmarks[14].x, landmarks[14].y])

    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    mouth_height = np.linalg.norm(mouth_upper - mouth_lower)

    if mouth_height < mouth_width * 0.3 and mouth_width > 0.1: # 임계값 조절 필요
        return "Happy"
    elif mouth_height > mouth_width * 0.7 and mouth_width < 0.05: # 임계값 조절 필요
        return "Sad"
    return None

# 웹캠 설정 (실시간 추적)
cap = cv2.VideoCapture(0)
pTime = 0

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
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
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

                landmarks = face_landmarks.landmark
                if landmarks:
                    eyebrow_emotion = calculate_eyebrow_expression(landmarks)
                    mouth_emotion = calculate_mouth_expression(landmarks)

                    if eyebrow_emotion:
                        emotion = eyebrow_emotion
                    elif mouth_emotion:
                        emotion = mouth_emotion
                    else:
                        emotion = "Neutral"

                cv2.putText(image, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Real-time Emotion Tracker (Basic)', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()