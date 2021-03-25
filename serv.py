import socket
import cv2
import numpy as np

import my_detect

print('ready to connect', flush=True)
sock = socket.socket()
sock.bind(('', 9878))
sock.listen(1)
conn, addr = sock.accept()

print('connected:', addr)

#l = int(input("Length of image"))
l = conn.recv(4)
l = int.from_bytes(l, 'little')
print(l)
chunk = 4096
read_by_now = 0
img_bytes = b''
while read_by_now < l - chunk:
    data = conn.recv(chunk)
    if not data:
        break
    img_bytes += data
    read_by_now += chunk
data = conn.recv(l - read_by_now)
img_bytes += data
print('received data')

img_bytes = np.array(bytearray(img_bytes))
img = cv2.imdecode(img_bytes, 1)
print('start to detect')
with open('out.png', 'wb') as fout:
    fout.write(img_bytes)
print('fouted successfully')
bboxes = my_detect.get_bboxes(img)
print('detected successfully')

valid_tetections =  bboxes[3][0]
boxes, scores, classes = bboxes[0][0], bboxes[1][0], list(map(int, bboxes[2][0]))
conn.send(valid_tetections.tobytes())

conn.close()