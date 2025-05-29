import gradio as gr
import cv2
import numpy as np
from PIL import Image

def detect_hand(image):
    # PIL → OpenCV
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 피부색 범위 (간단 기준)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # 윤곽선
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)

    # OpenCV → PIL
    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)
    return result_img

demo = gr.Interface(
    fn=detect_hand,
    inputs=gr.Image(type="pil", label="손 이미지 업로드"),
    outputs=gr.Image(type="pil", label="윤곽선 결과"),
    title="🖐 손 윤곽선 인식기",
    description="업로드한 손 이미지에서 OpenCV로 피부색 기반 윤곽선을 감지합니다."
)

if __name__ == "__main__":
    demo.launch()
