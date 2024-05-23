import numpy as np
import cv2
import mediapipe as mp
from functions import draw_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    annotatedFrame = frame
    if results.multi_hand_landmarks:
        annotatedFrame = draw_landmarks(frame, results.multi_hand_landmarks)

    cv2.imshow('annotated video', annotatedFrame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
