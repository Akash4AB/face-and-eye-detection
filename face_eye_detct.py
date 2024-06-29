import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image

def detect_faces_eyes(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
    return frame
example_data = {"name": "Akash", "project": "Face and Eye Detection"}

# Save the example data to a pickle file
with open("data.pkl", "wb") as f:
    pickle.dump(example_data, f)


def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpaperset.com/w/full/6/7/3/169822.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Face and Eye Detection")
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose Input Type", ("Webcam", "Upload Video"))

    if option == "Webcam":
        start_webcam = st.sidebar.button("Start Webcam")
        stop_webcam = st.sidebar.button("Stop Webcam")
        if start_webcam:
            cap = cv2.VideoCapture(0)
            st.session_state['cap'] = cap

        if 'cap' in st.session_state and st.session_state['cap'].isOpened():
            cap = st.session_state['cap']
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture image")
                    break
                frame = detect_faces_eyes(frame)
                stframe.image(frame, channels="BGR")
                if stop_webcam:
                    cap.release()
                    del st.session_state['cap']
                    stframe.empty()
                    break

    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            video_bytes = uploaded_file.read()
            video_file = open("uploaded_video.mp4", "wb")
            video_file.write(video_bytes)
            video_file.close()

            cap = cv2.VideoCapture("uploaded_video.mp4")
            stframe = st.empty()
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 360))
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 == 0:
                    frame = detect_faces_eyes(frame)

                stframe.image(frame, channels="BGR")
                frame_count += 1
            cap.release()

    st.sidebar.info("Use 'Start Webcam' to begin webcam feed and 'Stop Webcam' to end it.")
    st.sidebar.markdown("## Connect with me (Akash AB)")
    st.sidebar.markdown(
        """
        <a href="https://github.com/Akash4AB" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width:24px; height:24px;"> GitHub
        </a>
        """, unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <a href="https://www.linkedin.com/in/akash-kumar-62ba8627a" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" style="width:24px; height:24px;"> LinkedIn
        </a>
        """, unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <a href="https://www.instagram.com/a.kash_ab?igsh=ZzJleDZweTN1NTJy" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/1384/1384063.png" alt="Instagram" style="width:24px; height:24px;"> Instagram
        </a>
        """, unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
