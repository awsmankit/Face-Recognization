import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Ensure the 'registrations' directory exists
os.makedirs("registrations", exist_ok=True)

# Dictionary for known face encodings
known_faces = {}

def load_face_detector():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

face_net = load_face_detector()

class DNNFaceDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img 
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return img

def record_attendance(name):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now()])

def save_new_face(image, name):
    filepath = f"registrations/{name}.jpg"
    cv2.imwrite(filepath, image)
    face = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(face)[0]
    known_faces[name] = encoding

st.title('Real-time Face Recognition with DNN')

with st.sidebar:
    st.header("Registration")
    name = st.text_input("Enter your name for registration:")
    register = st.button("Register Face")

media_stream_constraints = {
    "video": {
        "width": {"ideal": 640},  # Reduced from HD/FHD to a lower resolution
        "height": {"ideal": 480},
        "frameRate": {"ideal": 30, "min": 15}  # Adjust frame rate as needed
    },
    "audio": False
}

ctx = webrtc_streamer(key="dnn_face_detection", video_processor_factory=DNNFaceDetector,
                      rtc_configuration=RTC_CONFIGURATION,
                      media_stream_constraints=media_stream_constraints)

if register and ctx.video_processor:
    if ctx.video_processor.last_frame is not None:
        img = ctx.video_processor.last_frame.to_ndarray(format="bgr24")
        save_new_face(img, name)
        st.success("Face registered successfully!")
