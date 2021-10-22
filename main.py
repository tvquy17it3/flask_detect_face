from flask import Flask
import sys
import time
import cv2
from facenet.face_contrib import *

app = Flask(__name__)

@app.route("/")
def index():
    face_recognition = Recognition('models', 'models/your_model.pkl')
    d = Detect(face_recognition)
    return "OK" + d
def add_overlays(faces, confidence=0.5):
    if faces is not None:
      for idx, face in enumerate(faces):
        face_bb = face.bounding_box.astype(int)
        if face.name and face.prob:
          if face.prob > confidence:
            return face.name
    return "False"

def Detect(face_recognition):
    frame= cv2.imread('quy.jpg')
    faces = face_recognition.identify(frame)
    return add_overlays(faces)

if __name__ == "__main__":
    app.run()
