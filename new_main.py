import os
import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import model_from_json
from model import load_eye_model, predict_eye

eyes_detected = defaultdict(str)
name = "kunal"

json_file = open(os.path.join('saved_models', 'emotion_model' , 'emotion_model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(os.path.join('saved_models', 'emotion_model', 'emotion_model.h5'))

cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def isBlinking(history, maxFrames):
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

eye_model = load_eye_model()

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = os.path.join('face_detection', 'harr_cascade', 'haarcascade_frontalface_alt.xml')
    open_eye_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_eye_tree_eyeglasses.xml')
    left_eye_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_lefteye_2splits.xml')
    right_eye_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_righteye_2splits.xml')

    face_detector = cv2.CascadeClassifier(facecasc)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        eyes = []
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # if open_eyes_glasses detect eyes then they are open
        if len(open_eyes_glasses) == 2:
            eyes_detected[name] += '1'
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # otherwise try detecting eyes using left and right_eye_detector
        # which can detect open and closed eyes
        else:
            # separate the face into left and right sides
            left_face = frame[y:y+h, x+int(w/2):x+w]
            left_face_gray = gray[y:y+h, x+int(w/2):x+w]

            right_face = frame[y:y+h, x:x+int(w/2)]
            right_face_gray = gray[y:y+h, x:x+int(w/2)]

            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            eye_status = '1'  # we suppose the eyes are open

            # For each eye check wether the eye is closed.
            # If one is closed we conclude the eyes are closed
            for (ex, ey, ew, eh) in right_eye:
                color = (0, 255, 0)
                pred = predict_eye(right_face[ey:ey+eh, ex:ex+ew], eye_model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(right_face, (ex, ey), (ex+ew, ey+eh), color, 2)
            for (ex, ey, ew, eh) in left_eye:
                color = (0, 255, 0)
                pred = predict_eye(left_face[ey:ey+eh, ex:ex+ew], eye_model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(left_face, (ex, ey), (ex+ew, ey+eh), color, 2)
            eyes_detected[name] += eye_status
        
        if isBlinking(eyes_detected[name], 3):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, name + 'real', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()