import sys
import time
import cv2
from facenet.face_contrib import *

def main():
  def add_overlays(faces, confidence=0.5):
    if faces is not None:
      for idx, face in enumerate(faces):
        face_bb = face.bounding_box.astype(int)
        if face.name and face.prob:
          if face.prob > confidence:
            class_name = face.name
            return True
    return False

  def HomeGUI():
    face_recognition = Recognition('models', 'models/your_model.pkl')
    frame= cv2.imread('quy.jpg')
    faces = face_recognition.identify(frame)
    return add_overlays(faces)

  d = HomeGUI()
  print(d)
if __name__ == '__main__':
  main()
