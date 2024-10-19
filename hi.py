import cv2
import numpy as np
import tensorflow_hub as hub
from string import ascii_uppercase


model = hub.KerasLayer(
    "https://www.kaggle.com/models/sayannath235/american-sign-language/TensorFlow2/american-sign-language/1"
)


def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)

    prediction = model(processed_frame)
    predicted_class = np.argmax(prediction)

    predicted_letter = (ascii_uppercase + "?" * 100)[predicted_class]

    cv2.putText(
        frame,
        str(chr(int(predicted_class))),
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("ASL Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
