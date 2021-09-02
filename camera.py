import numpy as np
import cv2
import concurrent.futures
from PIL import Image
import time


import core.utils as utils
import cProfile

def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper

def take_shot_and_detect(cap):
    import my_detect
    t0 = time.time()
    ret, frame = cap.read()
    if ret:
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bboxes = my_detect.get_bboxes(frame)
        frame = utils.draw_bbox(frame, bboxes)
        #frame = Image.fromarray(frame.astype(np.uint8))
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',frame)
        t = time.time() - t0
        print(bboxes[-1][0], 'boxes, time:', t)

@profile
def main_camera():
    cap = cv2.VideoCapture(0)
    t0 = time.time()
    bboxes = None
    current_frame = 0
    #with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    while(True):
        #executor.submit(take_shot_and_detect)
        take_shot_and_detect(cap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        current_frame += 1
        # if current_frame > 20:
            # break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

main_camera()