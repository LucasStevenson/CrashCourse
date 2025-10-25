import socket
import json
import struct
import numpy as np
import cv2

# Constants
HOST = 'localhost'
PORT = 55000
HEADER_SIZE = 8  # 4 bytes for data type, 4 bytes for payload length
IMAGE_BUFFER_SIZE = 4096

def receive_all(conn, length):
    """Helper to receive exact number of bytes."""
    # We need this func bcuz recv(n) is not always guaranteed to return exactly n bytes
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return bgr

def grade_driving():
    """Mock function to process frames."""
    # make request to toolhouse api for final result
    return {"grade": "A", "confidence": 0.97}

# Create TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"Listening on {HOST}:{PORT}")

conn, addr = server.accept()
print(f"Connection from {addr}")

image_frames = []

try:
    while True:
        header = receive_all(conn, HEADER_SIZE)
        if not header:
            break

        # Unpack the header: first 4 bytes = type, next 4 = length
        msg_type_bytes, msg_len_bytes = header[:4], header[4:]
        msg_type = msg_type_bytes.decode('utf-8').strip()
        msg_len = struct.unpack('!I', msg_len_bytes)[0]

        payload = receive_all(conn, msg_len)
        if not payload:
            break

        if msg_type == 'IMG':
            img = decode_image(payload)
            image_frames.append(img)
        elif msg_type == 'JSON':
            data = json.loads(payload.decode('utf-8'))
            print("Received JSON:", data)
        elif msg_type == 'DONE':
            print("Processing frames...")
            result = grade_driving()  # example: last frame
            conn.send(json.dumps(result).encode('utf-8'))
            image_frames.clear()
        else:
            print("Unknown message type:", msg_type)

finally:
    conn.close()
    server.close()

## Example request to send image
"""
header = b'IMG ' + struct.pack('!I', len(image_bytes))
sock.sendall(header + image_bytes)
"""
