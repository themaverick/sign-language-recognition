import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        # Calculate bounding box
        padding_box = 0.2 #The amount of extra space in the bounding box (20% more than the actual boundary)


        b_box_x = min(landmark.x for landmark in hand_landmarks.landmark)
        b_box_y = min(landmark.y for landmark in hand_landmarks.landmark)
        b_box_width = (max(landmark.x for landmark in hand_landmarks.landmark) - b_box_x)
        box_width = (1 + padding_box)*b_box_width
        b_box_height = (max(landmark.y for landmark in hand_landmarks.landmark) - b_box_y)
        box_height = (1 + padding_box)*b_box_height
        box_x = b_box_x - (padding_box/2)*b_box_width
        box_y = b_box_y - (padding_box/2)*b_box_height
        # Draw bounding box 
        #As all the values box_x, box_y are normalized, to get the real values for them we need to multiply them with their respective frame dimension
        cv2.rectangle(frame, (int(box_x * frame.shape[1]), int(box_y * frame.shape[0])),
                        (int((box_x + box_width) * frame.shape[1]), int((box_y + box_height) * frame.shape[0])),
                        (0, 255, 0), 2)


    cv2.imshow('annotated image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
