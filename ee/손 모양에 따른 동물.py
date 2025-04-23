import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

def recognize_animal(hand_landmarks):
    """간단한 손 랜드마크 분석으로 동물을 인식하는 함수 (미완성)"""
    if hand_landmarks:
        # 엄지 손가락과 다른 손가락들의 상대적인 위치를 분석하여
        # 특정 손 모양을 감지하고 그에 맞는 동물을 반환합니다.

        # **주의:** 이 부분은 매우 단순화된 예시이며, 실제 손 모양 인식은 훨씬 복잡합니다.
        thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        middle_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                              hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
        ring_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y])
        pinky_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y])
        wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                          hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])

        # 예시: 주먹 모양 (정확한 구현 필요)
        if index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and \
           ring_tip[1] < wrist[1] and pinky_tip[1] < wrist[1] and thumb_tip[0] < wrist[0]:
            return "곰"

        # 예시: 펼친 손 (정확한 구현 필요)
        elif abs(index_tip[1] - pinky_tip[1]) < 0.1 and index_tip[1] < wrist[1]:
            return "새"

        # 더 많은 손 모양-동물 연결 규칙 추가 필요

    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image_bgr.shape

    animal_name = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            animal_name = recognize_animal(hand_landmarks)

    if animal_name:
        cv2.putText(image_bgr, animal_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand to Animal', cv2.flip(image_bgr, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()