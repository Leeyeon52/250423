import cv2
import mediapipe as mp
import time
import numpy as np

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 🐶 동물 이미지 불러오기 (배경 투명 PNG)
animal_img = cv2.imread('dog.png', cv2.IMREAD_UNCHANGED)  # 반드시 4채널 PNG

# 웹캠 설정
cap = cv2.VideoCapture(0)
pTime = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose, \
    mp_face.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        pose_results = pose.process(image_rgb)
        face_results = face_detection.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 얼굴 위에 동물 이미지 씌우기
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_width = int(bbox.width * w)
                box_height = int(bbox.height * h)

                # 동물 이미지 resize
                resized_animal = cv2.resize(animal_img, (box_width, box_height))

                # 위치 조정
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + box_width), min(h, y + box_height)
                animal_roi = resized_animal[0:(y2 - y1), 0:(x2 - x1)]

                # 알파 채널 분리
                if animal_roi.shape[2] == 4:
                    alpha_s = animal_roi[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    for c in range(3):
                        image[y1:y2, x1:x2, c] = (
                            alpha_s * animal_roi[:, :, c] +
                            alpha_l * image[y1:y2, x1:x2, c]
                        )

        # 포즈 그리기
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

        # FPS 표시
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Pose with Animal Face', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
