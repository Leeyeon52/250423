import requests

def predict_image(image_path):
    print(image_path)
    url = f"http://127.0.0.1:8080/?img={image_path}"
    
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print("Error:", response.json())

# Example usage:
predict_image('./label_0_img_2.png')