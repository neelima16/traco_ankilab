import cv2
import json
import matplotlib.pyplot as plt

# Path to the video file
video_file = 'test/test001.mp4'

# Read the first frame of the video
cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()

# Path to the JSON file containing positions
json_file = 'test/test001.traco'

# Load positions from JSON
with open(json_file, 'r') as f:
    data = json.load(f)
    positions = [roi['pos'] for roi in data['rois'] if roi['z'] == 0]

# Overlay dots on the first frame
for x, y in positions:
    cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), -1)

# Display the frame with dots
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
