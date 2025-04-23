import cv2
import numpy as np

# 추적할 보라색 범위 (HSV) 정의
lower_purple = np.array([120, 80, 80])
upper_purple = np.array([160, 255, 255])

# 웹캠 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 이미지를 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 보라색 마스크 생성
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # 마스크에서 객체의 중심 찾기
    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        # 중심에 원 그리기 (초록색으로 변경)
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)

    # 결과 화면 보여주기
    cv2.imshow("Purple Color Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()