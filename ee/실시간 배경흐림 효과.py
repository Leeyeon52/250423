import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (192, 192, 192) # 흐려진 배경색 (선택 사항)

# SelfiSegmentation 모델 로드
with mp_selfie_segmentation.SelfiSegmentation(
    model_selection=0) as selfie_segmentation:
    # 웹캠 시작
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 성능 향상을 위해 이미지를 쓰기 불가능으로 표시
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 배경 흐림 효과 적용
        if results.segmentation_mask is not None:
            segmentation_mask = np.expand_dims(results.segmentation_mask, axis=-1)
            condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1

            # 배경 흐림 처리 (가우시안 블러)
            blurred_image = cv2.GaussianBlur(image, (55, 55), 0)

            # 배경을 흐려진 이미지로, 사람 영역은 원본 이미지로 합성
            output_image = np.where(condition, image, blurred_image)
        else:
            output_image = image

        # 결과 화면 보여주기
        cv2.imshow('Selfie Segmentation', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()