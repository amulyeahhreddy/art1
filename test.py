import cv2
import mediapipe as mp
import sys
import time

print("Starting...")
sys.path.insert(0, '.')
from mudra_recognizer import MudraRecognizer
print("Imports done.")

r = MudraRecognizer()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

poses = [
    ('HAMSASYA - thumb+index touch, middle+ring+pinky UP', 5),
    ('KATAKAMUKHA - thumb+index+middle triangle, ring+pinky UP', 5),
    ('ARALA - index curved, middle+ring+pinky straight', 5),
    ('SHUKATUNDA - index curved + ring bent, middle+pinky straight', 5),
]

for pose_name, duration in poses:
    print(f'\n>>> Hold {pose_name} <<<')
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            lm = [(l.x, l.y, l.z)
                  for l in results.multi_hand_landmarks[0].landmark]
            hs = r._hand_size(lm)
            ia = r._fangle(lm, 5, 6, 8)
            ma = r._fangle(lm, 9, 10, 12)
            ra = r._fangle(lm, 13, 14, 16)
            pa = r._fangle(lm, 17, 18, 20)
            pinch = r._dist(lm[4], lm[8]) / hs
            mid_ext = r._extended(lm, 9, 10, 12, 138)
            ring_ext = r._extended(lm, 13, 14, 16, 138)
            ring_bent = r._bent(lm, 13, 14, 16, 128)
            pink_bent = r._bent(lm, 17, 18, 20, 128)
            mudra, score = r.recognize_single(lm, 'Right')
            groups, _ = r._get_groups(lm, hs)
            curl = r._index_curl(lm)
            print(
                f'  {mudra}({score:.2f}) | '
                f'idx:{ia:.0f} mid:{ma:.0f} '
                f'ring:{ra:.0f} pink:{pa:.0f} | '
                f'pinch:{pinch:.2f} | '
                f'mid_ext:{mid_ext} ring_ext:{ring_ext} '
                f'ring_bent:{ring_bent} pink_bent:{pink_bent} | '
                f'groups:{groups}'
            )
            print(f'  INDEX CURL: {curl:.3f} (straight=1.0, curled=0.0-0.6)')
        time.sleep(0.5)

cap.release()
print('\nDone.')