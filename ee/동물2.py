import cv2
import mediapipe as mp
import random
import time
import numpy as np

# 이미지 파일 경로 설정
animal_images = {
    'dog': cv2.imread('images/dog.png', cv2.IMREAD_UNCHANGED),  # 알파 채널을 포함한 이미지 로드
    'cat': cv2.imread('images/cat.png', cv2.IMREAD_UNCHANGED),
    'panda': cv2.imread('images/panda.png', cv2.IMREAD_UNCHANGED),
    'fox': cv2.imread('images/fox.png', cv2.IMREAD_UNCHANGED),
    'dino': cv2.imread('images/dino.png', cv2.IMREAD_UNCHANGED),
    'lion': cv2.imread('images/lion.png', cv2.IMREAD_UNCHANGED),  # lion 이미지 추가
}

# 미디어파이프 손 모델
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 손가락 상태 확인 함수 (1: 펴짐, 0: 접힘)
def fingers_up(landmarks):
    fingers = []

    # 엄지
    fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    # 검지~새끼
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

    return fingers

# 제스처 분류 함수
def classify_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "rock"  # 주먹
    elif fingers[1:3] == [1, 1] and fingers[3:] == [0, 0]:
        return "scissors"  # 가위
    elif fingers == [1, 1, 1, 1, 1]:
        return "paper"  # 보
    else:
        return "none"

# 동물이 손에 나타나도록 하는 함수
def show_animal_on_hand(frame, landmarks, animal_image):
    # 손 위치를 기준으로 동물 이미지를 조정
    h, w, _ = frame.shape
    hand_x = int((landmarks[9].x + landmarks[0].x) * w / 2)  # 손목에서 중지까지의 x 위치
    hand_y = int((landmarks[9].y + landmarks[0].y) * h / 2)  # 손목에서 중지까지의 y 위치

    animal_h, animal_w, _ = animal_image.shape

    # 동물 이미지 크기 조정
    scale = 0.15  # 동물 이미지 크기 비율
    animal_image_resized = cv2.resize(animal_image, (int(animal_w * scale), int(animal_h * scale)))

    # 알파 채널 처리: 알파 채널이 있을 경우에만 처리
    if animal_image_resized.shape[2] == 4:
        alpha_s = animal_image_resized[:, :, 3] / 255.0  # 알파 채널
    else:
        alpha_s = np.ones(animal_image_resized.shape[:2])  # 알파 채널이 없다면 완전 불투명

    # 동물 이미지가 손 위치에 맞도록 자르기
    roi = frame[hand_y:hand_y + animal_image_resized.shape[0], hand_x:hand_x + animal_image_resized.shape[1]]

    # 동물 이미지를 손에 맞게 합성
    for c in range(0, 3):
        roi[:, :, c] = (alpha_s * animal_image_resized[:, :, c] + (1 - alpha_s) * roi[:, :, c])

    frame[hand_y:hand_y + animal_image_resized.shape[0], hand_x:hand_x + animal_image_resized.shape[1]] = roi

    return frame

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 동물 그룹 정의 (주먹, 가위, 보에 따라 다르게 나타날 동물들)
animal_groups = {
    "rock": ["dog", "fox"],
    "scissors": ["cat", "panda"],
    "paper": ["dino", "lion"],
}

selected_animal = None
last_change_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # RGB로 변환 (OpenCV는 BGR, Mediapipe는 RGB 사용)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손가락 상태 확인
            fingers = fingers_up(hand_landmarks.landmark)

            # 제스처 분류
            gesture = classify_gesture(fingers)

            # 제스처에 맞는 동물 랜덤 선택
            if gesture in animal_groups:
                if random.random() < 0.02:  # 너무 자주 바뀌지 않게
                    selected_animal = random.choice(animal_groups[gesture])

                # 동물 이미지가 None인 경우 예외 처리
                if selected_animal in animal_images and animal_images[selected_animal] is not None:
                    frame = show_animal_on_hand(frame, hand_landmarks.landmark, animal_images[selected_animal])
                else:
                    print(f"Error: {selected_animal} image is missing or invalid.")
            
            # 손 랜드마크를 그리기 (디버깅용)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ESC 키 눌렀을 때 종료
    cv2.imshow("Animal Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 27은 ESC 키
        break

cap.release()
cv2.destroyAllWindows()
