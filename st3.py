import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import cv2

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 실시간 비디오 처리 클래스 정의
class HandLandmarkProcessor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=2,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return img

# Streamlit 앱
st.title("🖐 실시간 손 랜드마크 인식")
st.write("웹캠을 통해 손가락 관절 위치를 실시간으로 인식합니다.")

webrtc_streamer(key="hand-detection", video_processor_factory=HandLandmarkProcessor)
