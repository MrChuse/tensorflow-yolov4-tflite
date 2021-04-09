import numpy as np
import cv2
import concurrent.futures
from PIL import Image
import time

import my_detect
import core.utils as utils

def take_shot_and_detect():
    global cap
    t0 = time.time()
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bboxes = my_detect.get_bboxes(frame)
    frame = utils.draw_bbox(frame, bboxes)
    frame = Image.fromarray(frame.astype(np.uint8))
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
    cv2.imshow('frame',frame)
    t = time.time() - t0
    print('time:', t)

cap = cv2.VideoCapture(0)
t0 = time.time()
bboxes = None
#with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
while(True):
    #executor.submit(take_shot_and_detect)
    take_shot_and_detect()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()