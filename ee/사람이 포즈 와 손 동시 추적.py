import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# 포즈 모델 초기화
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_poses=2)

# 손 모델 초기화
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
pTime = 0

def is_fist(hand_landmarks):
    # 엄지, 검지, 중지, 약지, 소지 끝의 y 좌표가 손바닥 근처 y 좌표보다 작으면 주먹으로 간주 (간단한 예시)
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        return thumb_tip > wrist_y and index_tip > wrist_y and middle_tip > wrist_y and ring_tip > wrist_y and pinky_tip > wrist_y
    return False

with pose, hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image_bgr.shape

        # 포즈 랜드마크 그리기
        if results_pose.multi_pose_landmarks:
            for i, pose_landmarks in enumerate(results_pose.multi_pose_landmarks):
                color = (0, 255 * (i + 1) % 256, 255 * i % 256)
                mp_drawing.draw_landmarks(
                    image_bgr,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style())

                # 두 사람이 감지되었을 경우 손목 사이 거리 측정
                if len(results_pose.multi_pose_landmarks) == 2:
                    person1_landmarks = results_pose.multi_pose_landmarks[0].landmark
                    person2_landmarks = results_pose.multi_pose_landmarks[1].landmark

                    # 오른쪽 손목 거리
                    right_wrist1 = np.array([person1_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                                             person1_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
                    right_wrist2 = np.array([person2_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                                             person2_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
                    distance_right_wrists = np.linalg.norm(right_wrist1 - right_wrist2)
                    cv2.putText(image_bgr, f'Dist(R): {int(distance_right_wrists)}',
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # 왼쪽 손목 거리
                    left_wrist1 = np.array([person1_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                                            person1_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
                    left_wrist2 = np.array([person2_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                                            person2_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h])
                    distance_left_wrists = np.linalg.norm(left_wrist1 - left_wrist2)
                    cv2.putText(image_bgr, f'Dist(L): {int(distance_left_wrists)}',
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 손 랜드마크 그리기 및 주먹 인식
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_fist(hand_landmarks):
                    # 손 랜드마크의 중간점 근처에 "Fist" 표시
                    hand_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w)
                    hand_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)
                    cv2.putText(image_bgr, "Fist", (hand_center_x - 30, hand_center_y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image_bgr, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Pose and Hand Tracking', cv2.flip(image_bgr, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()