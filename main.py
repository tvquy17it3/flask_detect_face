from flask import Flask, request
from flask_cors import CORS, cross_origin
from facenet.face_contrib import *
import sys
import time
import datetime
import cv2
import base64
import numpy as np
# python3.7 -m pip install package

sess = tf.Session()
graph = tf.get_default_graph()

with sess.as_default():
  with graph.as_default():
    face_recognition = Recognition('models', 'models/your_model.pkl')


app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#=========================================================
def base64_image(base64_img):
    try:
        base64_img = np.fromstring(base64.b64decode(base64_img), dtype=np.uint8)
        base64_img = cv2.imdecode(base64_img, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return base64_img

def add_overlays(faces, user_id, confidence=0.5):
  if faces is not None:
    for idx, face in enumerate(faces):
      face_bb = face.bounding_box.astype(int)
      if face.name and face.prob:
        if face.prob > confidence:
          if user_id == face.name:
            conf = '{:.02f}'.format(face.prob * 100)
            return True, conf
  return False, 0
#=========================================================

@app.route("/")
def wellcome():
  return "Wellcome!"
#==========================================================

@app.route('/detect', methods=['POST'])
@cross_origin(origin='*')
def index():
  data = request.json
  image = base64_image(data['image'])
  if image is None:
    return {"data": False, "message": "Cannot load image"}
  else:
    with sess.as_default():
      with graph.as_default():
        faces = face_recognition.identify(image)
        user_id = str(data['user_id'])
        check_confidence, conf = add_overlays(faces, user_id)
        if check_confidence:
          time_now = time.time()
          date_time = datetime.datetime.fromtimestamp(time_now).strftime('%d-%m-%Y %H:%M:%S')
          return {
                  "data": True,
                  "conf": conf,
                  "date_time": str(date_time),
                  "user_id": user_id,
                }

  return {"data": False, "message": "Cannot detect"}
#==========================================================

if __name__ == "__main__":
  app.run(host='0.0.0.0')

# url= "http://127.0.0.1:9000/predict"
# post_data = json.dumps({'month': month, 'day': day, 'hour': hour,})
# r = rq.post(url,post_data)
#face_recognition = Recognition('models', 'models/your_model.pkl')
