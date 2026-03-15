import cv2
import numpy as np
from mandala_renderer import MandalaRenderer

mandala = MandalaRenderer(width=640, height=480)
mandala.set_mudra('Pataka')
print("Press 1-5 to switch patterns, q to quit")
print("1=Lotus 2=Yantra 3=Floral 4=Spiral 5=Star")

while True:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mandala.update()
    mandala.render(frame)
    cv2.imshow('Mandala Test', frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        mandala.set_mudra('Pataka')
        print("Lotus")
    elif key == ord('2'):
        mandala.set_mudra('Mushti')
        print("Yantra")
    elif key == ord('3'):
        mandala.set_mudra('Arala')
        print("Floral")
    elif key == ord('4'):
        mandala.set_mudra('Mayura')
        print("Spiral")
    elif key == ord('5'):
        mandala.set_mudra('Suchi')
        print("Star")

cv2.destroyAllWindows()
