import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from imutils.video import VideoStream
from model import load_eye_model, predict_eye


def isBlinking(history, maxFrames):
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False


def init():
    # eye detection libraries
    face_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_frontalface_alt.xml')
    open_eye_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_eye_tree_eyeglasses.xml')
    left_eye_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_lefteye_2splits.xml')
    right_eye_cascPath = os.path.join('face_detection', 'harr_cascade','haarcascade_righteye_2splits.xml')

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    print("[LOG] Opening webcam ...")
    video_capture = VideoStream(src=0).start()

    model = load_eye_model()

    return (model, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, video_capture)


def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, eyes_detected):
    name = 'person1'
    frame = video_capture.read()
    # resize the frame
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # for each detected face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]

        eyes = []
        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # if open_eyes_glasses detect eyes then they are open
        if len(open_eyes_glasses) == 2:
            eyes_detected[name] += '1'
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

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
                pred = predict_eye(right_face[ey:ey+eh, ex:ex+ew], model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(right_face, (ex, ey), (ex+ew, ey+eh), color, 2)
            for (ex, ey, ew, eh) in left_eye:
                color = (0, 255, 0)
                pred = predict_eye(left_face[ey:ey+eh, ex:ex+ew], model)
                if pred == 'closed':
                    eye_status = '0'
                    color = (0, 0, 255)
                cv2.rectangle(left_face, (ex, ey), (ex+ew, ey+eh), color, 2)
            eyes_detected[name] += eye_status

        # Each time, we check if the person has blinked
        # If yes, we display its name
        if isBlinking(eyes_detected[name], 3):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Display name
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, name + 'real', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame


if __name__ == "__main__":
    (model, face_detector, open_eyes_detector, left_eye_detector,
     right_eye_detector, video_capture) = init()
    eyes_detected = defaultdict(str)
    while True:
        frame = detect_and_display(model, video_capture, face_detector,
                                   open_eyes_detector, left_eye_detector, right_eye_detector, eyes_detected)
        cv2.imshow("Face Liveness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_capture.stop()
