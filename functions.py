import cv2
import mediapipe as mp
import numpy as np
import pickle
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# to calculate the slope of a 2d vector
def slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)

# to calculate the angle between two vectors from their slope.
def angle(m1, m2):
    math.degrees(math.atan(np.abs((m2 - m1)/(1 + m1*m2))))


# fold ratio is the ratio of the length of the segments of the finger to the distance between the nuckle to the tip of the finger.
def foldRatio(arr):
    i, j, k = arr.shape
    fingVecs = np.zeros((i, j-1, k))

    for l in range(j-1):
        fingVecs[:, l] = arr[:, l+1] - arr[:, l]

    lenFing = (np.sqrt(((fingVecs**2).sum(axis = -1)))).sum(-1)

    #extreme distance
    extDist = np.sqrt(((arr[:, -1] - arr[:, 0])**2).sum(-1))

    ratio = (extDist/lenFing)
    return ratio

# to calculate the unit vectors of the segments of the fingers.
def fingVec(arr):
    #shape = (5, 4, 3)
    i, j, k = arr.shape
    #shape = (5, 3, 3)
    fingVecs = np.zeros((i, j-1, k))

    for l in range(j-1):
        fingVecs[:, l] = arr[:, l+1] - arr[:, l]

    #shape = (5, 3, 3)
    fingVecMag = np.sqrt(np.expand_dims(((fingVecs**2).sum(axis = -1)), axis = -1))
    fingUnVecs = fingVecs/fingVecMag

    return fingUnVecs

# to extract finger vectors from the landmarks.
def extrFing(landmarks):
    for hand_landmarks in landmarks:
        listLmk = hand_landmarks.landmark

        #fingers index, middle, ring, pinky, thumb
        fings = np.array([[[listLmk[5].x, listLmk[5].y, listLmk[5].z], [listLmk[6].x, listLmk[6].y, listLmk[6].z], [listLmk[7].x, listLmk[7].y, listLmk[7].z], [listLmk[8].x, listLmk[8].y, listLmk[8].z]], 
                            [[listLmk[9].x, listLmk[9].y, listLmk[9].z],[listLmk[10].x, listLmk[10].y, listLmk[10].z], [listLmk[11].x, listLmk[11].y, listLmk[11].z],  [listLmk[12].x, listLmk[12].y, listLmk[12].z]], 
                            [[listLmk[13].x, listLmk[13].y, listLmk[13].z],[listLmk[14].x, listLmk[14].y, listLmk[14].z], [listLmk[15].x, listLmk[15].y, listLmk[15].z],  [listLmk[16].x, listLmk[16].y, listLmk[16].z]], 
                            [[listLmk[17].x, listLmk[17].y, listLmk[17].z],[listLmk[18].x, listLmk[18].y, listLmk[18].z], [listLmk[19].x, listLmk[19].y, listLmk[19].z],  [listLmk[20].x, listLmk[20].y, listLmk[20].z]], 
                            [[listLmk[1].x, listLmk[1].y, listLmk[1].z],[listLmk[2].x, listLmk[2].y, listLmk[2].z], [listLmk[3].x, listLmk[3].y, listLmk[3].z],  [listLmk[4].x, listLmk[4].y, listLmk[4].z]]])
    return fings





    

                