#pip install opencv-python mediapipe

import cv2
import mediapipe as mp
import numpy as np

video = cv2.VideoCapture(0)

hand = mp.solutions.hands
Hands = hand.Hands(max_num_hands=2, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    if not check:
        break

    img = cv2.flip(img, 1)
    total_dedos = 0

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints:
        for hand_landmarks, handedness in zip(handsPoints, results.multi_handedness):
            pontos = []

            #mpDraw.draw_landmarks(img, hand_landmarks, hand.HAND_CONNECTIONS)
            
            for id, cord in enumerate(hand_landmarks.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                #Se quiser ver os números da marcação da mão, descomente a linha abaixo
                #cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                pontos.append((cx, cy))

            hand_label = handedness.classification[0].label

            wrist_coords = tuple(
                np.multiply([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                             hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y],
                            [w, h]).astype(int))
            #Se quiser ver se a mão é direita ou esquerda descomente a linha abaixo
            #cv2.putText(img, hand_label, wrist_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            dedos = [8, 12, 16, 20]
            contador = 0
            if pontos:
                if hand_label == "Right":
                    if pontos[4][0] < pontos[2][0]:
                        contador += 1
                if hand_label == "Left":
                    if pontos[4][0] > pontos[2][0]:
                        contador += 1
                for x in dedos:
                    if pontos[x][1] < pontos[x - 2][1]:
                        contador += 1
                total_dedos += contador

        cv2.putText(img, f'Quantidade de dedos: {total_dedos}', (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    cv2.imshow("Imagem", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
Hands.close()
