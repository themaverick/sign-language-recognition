import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Function to draw landmarks as well as bounding box if needed
def draw_landmarks(frame, landmarks, bounding_box = True, padding = 0.2):
    # Drawing landmarks as well as the frame which are both available
    for hand_landmarks in landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
    
    if bounding_box:
        # Drawing the bounding box if selected so

        #calculating the dimensions of the box
        b_box_x = min(landmark.x for landmark in hand_landmarks.landmark)
        b_box_y = min(landmark.y for landmark in hand_landmarks.landmark)
        b_box_width = (max(landmark.x for landmark in hand_landmarks.landmark) - b_box_x)
        box_width = (1 + padding)*b_box_width
        b_box_height = (max(landmark.y for landmark in hand_landmarks.landmark) - b_box_y)
        box_height = (1 + padding)*b_box_height
        box_x = b_box_x - (padding/2)*b_box_width
        box_y = b_box_y - (padding/2)*b_box_height

        # Draw bounding box 
        #As all the values box_x, box_y are normalized, to get the real values for them we need to multiply them with their respective frame dimension
        cv2.rectangle(frame, (int(box_x * frame.shape[1]), int(box_y * frame.shape[0])),
                        (int((box_x + box_width) * frame.shape[1]), int((box_y + box_height) * frame.shape[0])),
                        (0, 255, 0), 2)
        
    return frame
