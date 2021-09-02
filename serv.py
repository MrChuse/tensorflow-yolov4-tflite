import socket
import cv2
import numpy as np
import threading
import time

import pickle

import my_detect

#PORT = 9001
PORT = 2233
IP = '' # all ips
POSSIBLE_CONNECTIONS = 1
CHUNK = 4096

with open('bboxes.dat', 'rb') as fin:
    bboxes = pickle.load(fin)

class Server:
    def __init__(self, ip, port):
        self.sock = socket.socket()
        self.sock.bind((ip, port))
        self.sock.listen(POSSIBLE_CONNECTIONS)
        print('Server Initialized')
        self.start_listener()
        
    def start_listener(self):
        # handles multiple connections and starts a thread for each of them
        try:
            connections = 0
            while True:
                print('Waiting for a connection...')
                conn, addr = self.sock.accept()
                connections += 1
                print('Connected!')
                t = threading.Thread(target=self.handle_connection, args=(connections, conn, addr), daemon=True)
                t.start()
        except TypeError as e:
            print('Exception:', e)
    
    def receive(self, conn, img_len, chunk):
        # wrapper function for conn.recv
        img_bytes = b''
        while len(img_bytes) < img_len:
            if img_len - len(img_bytes) < chunk:
                data = conn.recv(img_len - len(img_bytes))
            else:
                data = conn.recv(chunk)
            if not data:
                break
            img_bytes += data
            #print('in receive', len(img_bytes), read_by_now)
        return img_bytes
    
    def convert_bytes_to_cv2_image(self, img_bytes):#, size=(640, 480)):
        # img_bytes = np.array(bytearray(img_bytes))
        # img = img_bytes.reshape((size[1], size[0], 3))
        # img = img[::-1, :, :]
        # img_bytes = img_bytes[np.mod(np.arange(img_bytes.size),4)!=3]
        img_bytes = np.array(bytearray(img_bytes))
        print('img_bytes in convert_cv2', img_bytes, img_bytes.shape)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        # print('img in convert_cv2', img, img.shape)
        return img
    
    def handle_connection(self, num, conn, addr):
        # implements a protocol for a connection
        print('Connection', num, ':', 'established')
        
        # width = int.from_bytes(conn.recv(4), 'little')
        # height = int.from_bytes(conn.recv(4), 'little')
        # print('Connection', num, ':', 'width and height:', width, height)
        
        cont_to_detect = int.from_bytes(conn.recv(4), 'little')
        while cont_to_detect:
            # main cycle to receive photos
            t0 = time.time()
            img_len = int.from_bytes(conn.recv(4), 'little')
            print('Connection', num, ':', img_len, 'bytes to receive')
            
            img_bytes = self.receive(conn, img_len, CHUNK)
            print('Connection', num, ':', 'received data')
            
            img = self.convert_bytes_to_cv2_image(img_bytes)#, (width, height))
            print('Connection', num, ':', 'converted image')
            
            print('Connection', num, ':', 'started detection')
            t1 = time.time()
            try:
                bboxes = my_detect.get_bboxes(img)
                #print('!!!!!img', img)
            except Exception as e:
                print('Exception!', e)
                print('exiting')
                conn.close()
                return
            t2 = time.time()
            print('Connection', num, ':', 'detected successfully')
            
            valid_detections =  bboxes[3][0]
            
            boxes = bboxes[0][0][:valid_detections].flatten()
            scores = bboxes[1][0][:valid_detections]
            classes = bboxes[2][0][:valid_detections].astype(int)
            # valid_detections = np.int32(1)
            # boxes = np.array([0.1, 0.2, 0.3, 0.4])
            # scores = np.array([1.])
            # classes = np.array([0]).astype(int)
            
            # print('valid', valid_detections, type(valid_detections))
            conn.send(valid_detections.tobytes())
            conn.send(bytes(scores))
            conn.send(bytes(classes))
            conn.send(bytes(boxes))
            print('scores', scores)
            # print('scores', len(bytes(scores)))
            # print('scores', bytes(scores))
            print('classes', classes)
            # print('classes', len(bytes(classes)))
            # print('classes', bytes(classes))
            # print('boxes, not flatten', bboxes[0][0][:valid_detections])
            print('boxes', boxes)
            # print('boxes', len(bytes(boxes)))
            # print('boxes', bytes(boxes))
            
            t3 = time.time()
            print('Connection', num, ':', 't_frame:', t3-t0,  ', t_detect:', t2-t1)
            cont_to_detect = int.from_bytes(conn.recv(4), 'little')
            print('continue?', cont_to_detect)
        print('Connection', num, ':', 'stopped connection and thread')
        conn.close()
        

def thread_function():
    # creates a Server
    myserver = Server(IP, PORT)

def main():
    # starts a thread with thread function
    t = threading.Thread(target=thread_function, daemon=True)
    t.start()
    print('Server Started...!')
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()
