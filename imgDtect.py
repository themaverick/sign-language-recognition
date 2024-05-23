import numpy as np
import cv2
import mediapipe as mp
from functions import draw_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

img = cv2.imread("./images/1.jpg")

rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = hands.process(rgb_frame)

annotatedImg = img
if results.multi_hand_landmarks:

    annotatedImg = draw_landmarks(img, results.multi_hand_landmarks)

        
cv2.imshow('annotated image', annotatedImg)
cv2.waitKey(0) 
cv2.destroyWindow("annotated image")
hands.close()

