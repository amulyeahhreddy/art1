import cv2
import mediapipe as mp
import sys
import time

sys.path.insert(0, '.')
from mudra_recognizer import MudraRecognizer

r = MudraRecognizer()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

print("Hold your pose. Printing for 8 seconds...")
print("=" * 60)

start = time.time()
while time.time() - start < 8:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = [(l.x, l.y, l.z) for l in results.multi_hand_landmarks[0].landmark]
        hs = r._hand_size(lm)
        groups, flags = r._get_groups(lm, hs)
        mudra, score = r.recognize_single(lm, 'Right')

        ie = r._extended(lm, 5, 6, 8, 138)
        me = r._extended(lm, 9, 10, 12, 138)
        re = r._extended(lm, 13, 14, 16, 138)
        pe = r._extended(lm, 17, 18, 20, 138)

        print(f"Mudra: {mudra} ({score:.2f})")
        print(f"Groups: {groups}")
        print(f"ie:{ie} me:{me} re:{re} pe:{pe}")
        print("-" * 40)
        time.sleep(0.5)

cap.release()
print("Done.")