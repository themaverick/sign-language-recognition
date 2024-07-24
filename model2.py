import numpy as np
import cv2
import mediapipe as mp
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

label = "*"
imgSize = 200
offset = 20
labels = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24}


class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, num_classes)


# Sequential Layer
  def forward(self, x):
    x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
    return x
  
model_1 = NN(63, 24)
model_1.load_state_dict(torch.load("./models/nnAbsLmks2"))
model_1.eval()

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
    res = hands.process(rgb_frame)
    

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            #for landmark in hand_landmarks.landmark:
            hll = hand_landmarks.landmark
            w = (max(lmk.x for lmk in hll) - min(lmk.x for lmk in hll))*rgb_frame.shape[1]
            h = (max(lmk.y for lmk in hll) - min(lmk.y for lmk in hll))*rgb_frame.shape[0]
            x = (min(lmk.x for lmk in hll)*rgb_frame.shape[1])
            y = (min(lmk.y for lmk in hll)*rgb_frame.shape[0])

            imgCrop = frame[int(y-offset):int(y+h+offset), int(x-offset):int(x+w+offset)]
        if imgCrop.size != 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio >+ 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            rgbFrame = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
            result = hands.process(rgbFrame)
            landmarks = []

            if result.multi_hand_landmarks:
                for handLandmarks in result.multi_hand_landmarks:
                    #for landmark in hand_landmarks.landmark:
                    HLL = handLandmarks.landmark
                    for s in range(21):
                        landmarks.append(HLL[s].x)
                        landmarks.append(HLL[s].y)
                        landmarks.append(HLL[s].z)
                landmarks = torch.from_numpy(np.array(landmarks)).float()
                if landmarks.shape[0] == 63:

                    with torch.no_grad():
                        out = np.argmax(model_1(landmarks).detach().numpy())

                        for k, v in labels.items():
                            if v == out:
                                label = k
    
    cv2.imshow("original video", frame)
    cv2.putText(imgWhite, label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("Captured Frame", imgWhite)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
