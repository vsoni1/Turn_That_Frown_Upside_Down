import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

emotion_model_path = '_mini_XCEPTION.hdf5'
emoji = cv2.imread('./happy.png', -1)

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_classifier = load_model(emotion_model_path, compile=False)
emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
cv2.namedWindow('face')

camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    frame_copy = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (X, Y, W, H) = faces
        roi = gray[Y:Y + H, X:X + W]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        label = emotions[preds.argmax()]
        if label not in ["happy", "surprised", "neutral"]:
            emoji = cv2.resize(emoji, (H,W))
            image_array = np.asarray(emoji)
            for c in range(0, 3):
                frame_copy[Y:Y + H, X:X + W, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                                    + frame_copy[Y:Y + H, X:X + W, c] * (1.0 - image_array[:, :, 3] / 255.0)
                
    cv2.imshow('face', frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
