import cv2

# .jpg 파일을 .png로 변환하는 함수
def convert_to_png(input_path, output_path):
    # 이미지 로드
    image = cv2.imread(input_path)
    
    # 이미지가 제대로 로드되었는지 확인
    if image is None:
        print(f"Error loading {input_path}")
        return
    
    # .png 형식으로 저장
    cv2.imwrite(output_path, image)
    print(f"Saved {output_path}")

# 이미지 경로
image_paths = [r'C:\Users\302-15\Desktop\ee\lion.jpg']
               

# 변환된 이미지 경로
output_paths = [r'C:\Users\302-15\Desktop\ee\lion.png']

# 각 이미지를 변환
for input_path, output_path in zip(image_paths, output_paths):
    convert_to_png(input_path, output_path)
