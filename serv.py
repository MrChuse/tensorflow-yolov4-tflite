import socket
import cv2
import numpy as np
import threading
import time

import my_detect

PORT = 9002
IP = '' # all ips
POSSIBLE_CONNECTIONS = 1
CHUNK = 4096

class Server:
    def __init__(self, ip, port):
        self.sock = socket.socket()
        self.sock.bind((ip, port))
        self.sock.listen(POSSIBLE_CONNECTIONS)
        print('Server Initialized')
        self.start_listener()
        
    def start_listener(self):
        try:
            connections = 0
            while True:
                print('Waiting for a connection...')
                conn, addr = self.sock.accept()
                connections += 1
                print('Connected!')
                t = threading.Thread(target=self.handle_connection, args=(connections, conn, addr))
                t.start()
        except TypeError as e:
            print('Exception:', e)
    
    def receive(self, conn, img_len, chunk):
        read_by_now = 0
        img_bytes = b''
        while read_by_now < img_len - chunk:
            data = conn.recv(chunk)
            if not data:
                break
            img_bytes += data
            read_by_now += chunk
        data = conn.recv(img_len - read_by_now)
        img_bytes += data
        return img_bytes
    
    def convert_bytes_to_cv2_image(self, img_bytes):
        img_bytes = np.array(bytearray(img_bytes))
        img = cv2.imdecode(img_bytes, 1)
        return img
    
    def handle_connection(self, num, conn, addr):
        # todo: add infinite loop
        print('Connection', num, ':', 'established')
        cont_to_detect = int.from_bytes(conn.recv(4), 'little')
        while cont_to_detect:
            t0 = time.time()
            img_len = int.from_bytes(conn.recv(4), 'little')
            print('Connection', num, ':', img_len, 'bytes to receive')
            
            img_bytes = self.receive(conn, img_len, CHUNK)
            print('Connection', num, ':', 'received data')
            
            img = self.convert_bytes_to_cv2_image(img_bytes)
            print('Connection', num, ':', 'converted image')
            
            print('Connection', num, ':', 'started detection')
            t1 = time.time()
            bboxes = my_detect.get_bboxes(img)
            t2 = time.time()
            print('Connection', num, ':', 'detected successfully')
            
            valid_tetections =  bboxes[3][0]
            boxes, scores, classes = bboxes[0][0], bboxes[1][0], list(map(int, bboxes[2][0]))
            conn.send(valid_tetections.tobytes())
            t3 = time.time()
            print('Connection', num, ':', 't_frame:', t3-t0,  ', t_detect:', t2-t1)
            cont_to_detect = int.from_bytes(conn.recv(4), 'little')
            print(cont_to_detect)
            cont_to_detect = 0
        print('Connection', num, ':', 'stopped connection and thread')
        conn.close()
        

def thread_function():
    myserver = Server(IP, PORT)

def main():
    t = threading.Thread(target=thread_function)
    t.start()
    print('Server Started...!')

if __name__ == '__main__':
    main()