import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from glob import glob

folder = "S:/Mark/Research/Fish Behavioural/30062017 Arnold Looming/Run2"
# filename = folder + "looming.csv"
# video = folder + '2017-06-27_15-24-13.mp4'
filename = glob(folder + "*.csv")[0]
video = glob(folder + "*.mp4")[0]

data = pd.read_csv(filename, header=None)

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
frame_rate = cap.get(cv2.CAP_PROP_FPS)
number_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.release()
video_length = number_frames / frame_rate

number_electrode = 10
plt.imshow(frame)
plt.title("Select Electrodes 1-10 (in order)")
electrodes = plt.ginput(n=number_electrode, timeout=0)
plt.close()

plt.imshow(frame)
plt.title("Select Indicator Led")
LED = plt.ginput(n=1, timeout=0)[0]
plt.close()

cap = cv2.VideoCapture(video)
led_values = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        val = gray[int(LED[1]), int(LED[0])]
        led_values.append(val)
    else:
        break

cap.release()
cv2.destroyAllWindows()
led_values = np.array(led_values)
plt.plot(led_values, 'o-')
plt.title('Pick Threshold')
threshold = plt.ginput(n=1, timeout=0)[0][1]
plt.close()
pks = np.where(led_values > threshold)[0]
splits = np.where(np.diff(pks) > 10)[0]
pks = pks[splits]


ix_start = pks[0]
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, ix_start)
i = 0
color_bounds = [200, 70000]
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        t = (i / frame_rate) % (data[0][len(data) - 1])
        d = data.ix[np.where(data[0] >= t)[0][0]][1:]
        for ix in np.arange(number_electrode):
            electrode_pos = electrodes[ix]
            d_ = d[ix + 1]
            color = (d_ - color_bounds[0]) / color_bounds[1] * 255
            cv2.circle(frame, (int(electrode_pos[0]), int(electrode_pos[1])), 5, (0, 255 - color, 0), -1)
        cv2.imshow('overlay', frame)
        i += 1
        wait = 1  # int(1000 / frame_rate)
        key = cv2.waitKey(wait) & 0xFF
        if key == ord('q'):
            break
    else:
        break
