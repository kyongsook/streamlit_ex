import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Hand Detector", layout="centered")
st.title("🖐 손 랜드마크 인식기")
st.write("손이 나온 이미지를 업로드하면 손가락 관절을 인식해 표시합니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # MediaPipe 설정
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        # 이미지 BGR로 변환 → MediaPipe 처리
        results = hands.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.image(img_np, caption="손 랜드마크 결과", use_column_width=True)
    st.success("✅ 분석 완료!")
