# Right now, this is only `pseudocode` for proof of concept
import socket

# Create TCP server, make it listen on port 55000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 55000))
server.listen(1)
conn, addr = server.accept()

image_frames = []

while True:
    # Receive image bytes from Unity
    image_bytes = conn.recv(IMAGE_BUFFER_SIZE)
    image_frames.append(decode_image(image_bytes))

# After we receive all the frames from Unity...
# Analyze image frames: run vision model, grade driving, etc.
result = grade_driving(image)
# Send feedback/results back to Unity
conn.send(result.encode('utf-8'))

