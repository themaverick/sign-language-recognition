import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
result, image = cam.read()
rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

results = hands.process(image)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks.landmark):
            print(f"Landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

    # Calculate bounding box
        b_box_x = min(landmark.x for landmark in hand_landmarks.landmark)
        b_box_y = min(landmark.y for landmark in hand_landmarks.landmark)
        b_box_width = max(landmark.x for landmark in hand_landmarks.landmark) - b_box_x
        b_box_height = max(landmark.y for landmark in hand_landmarks.landmark) - b_box_y
        # Draw bounding box
        cv2.rectangle(image, (int(b_box_x * image.shape[1]), int(b_box_y * image.shape[0])),
                        (int((b_box_x + b_box_width) * image.shape[1]), int((b_box_y + b_box_height) * image.shape[0])),
                        (0, 255, 0), 2)
        
cv2.imshow('annotated image', image)
cv2.waitKey(0) 
cv2.destroyWindow("annotated image")

