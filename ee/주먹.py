import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 주먹 쥐는 횟수 관련 변수
fist_count = 0
is_fist = False
thumb_tip_index_mcp_dist_threshold_ratio = 0.3  # 엄지 끝과 검지 손가락 뿌리 사이 거리 비율 (조절 필요)

def is_fist_formed(hand_landmarks, image_width, image_height):
    if not hand_landmarks:
        return False

    # 랜드마크 좌표 추출
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # 엄지 끝과 다른 손가락 뿌리 사이의 거리 계산
    dist_thumb_index = ((thumb_tip.x - index_finger_mcp.x)**2 + (thumb_tip.y - index_finger_mcp.y)**2)**0.5
    dist_thumb_middle = ((thumb_tip.x - middle_finger_mcp.x)**2 + (thumb_tip.y - middle_finger_mcp.y)**2)**0.5
    dist_thumb_ring = ((thumb_tip.x - ring_finger_mcp.x)**2 + (thumb_tip.y - ring_finger_mcp.y)**2)**0.5
    dist_thumb_pinky = ((thumb_tip.x - pinky_finger_mcp.x)**2 + (thumb_tip.y - pinky_finger_mcp.y)**2)**0.5

    # 손가락 끝이 손바닥 근처에 있는지 확인 (주먹 쥔 상태)
    palm_center_x = (index_finger_mcp.x + pinky_finger_mcp.x) / 2
    palm_center_y = (index_finger_mcp.y + pinky_finger_mcp.y) / 2

    dist_index_palm = ((index_finger_tip.x - palm_center_x)**2 + (index_finger_tip.y - palm_center_y)**2)**0.5
    dist_middle_palm = ((middle_finger_tip.x - palm_center_x)**2 + (middle_finger_tip.y - palm_center_y)**2)**0.5
    dist_ring_palm = ((ring_finger_tip.x - palm_center_x)**2 + (ring_finger_tip.y - palm_center_y)**2)**0.5
    dist_pinky_palm = ((pinky_finger_tip.x - palm_center_x)**2 + (pinky_finger_tip.y - palm_center_y)**2)**0.5

    # 임계값 비교 (조절 필요)
    fist_threshold = 0.2  # 손가락 끝과 손바닥 중심 사이 거리의 최대 비율

    return (dist_thumb_index < thumb_tip_index_mcp_dist_threshold_ratio and
            dist_thumb_middle < thumb_tip_index_mcp_dist_threshold_ratio and
            dist_thumb_ring < thumb_tip_index_mcp_dist_threshold_ratio and
            dist_thumb_pinky < thumb_tip_index_mcp_dist_threshold_ratio and
            dist_index_palm < fist_threshold and
            dist_middle_palm < fist_threshold and
            dist_ring_palm < fist_threshold and
            dist_pinky_palm < fist_threshold)

# 웹캠 설정 (실시간 추적)
cap = cv2.VideoCapture(0)
pTime = 0

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                h, w, _ = image.shape
                if is_fist_formed(hand_landmarks, w, h) and not is_fist:
                    is_fist = True
                elif not is_fist_formed(hand_landmarks, w, h) and is_fist:
                    fist_count += 1
                    is_fist = False
                    print(f"주먹 쥔 횟수: {fist_count}")

        else:
            is_fist = False # 손이 감지되지 않으면 주먹 상태 초기화

        cv2.putText(image, f'Fist Count: {fist_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Real-time Fist Tracker', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

python fist_tracker.py