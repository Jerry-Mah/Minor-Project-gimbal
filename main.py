import cv2
from ultralytics import YOLO
from pyfirmata import Arduino, SERVO
import numpy as np
import time

# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')
# model2 = YOLO('yolov8n.pt')

# Video parameters
video_path = "people.mp4"
capture_video = False
output_video_path = 'gallery/gogo.avi'




# port = 'COM5'
# board = Arduino(port)
# pin_10= 10
# pin_9 = 8
# board.digital[pin_10].mode = SERVO
# board.digital[pin_9].mode = SERVO

# servo_pinX = board.get_pin('d:9:s') #pin 9 Arduino
# servo_pinY = board.get_pin('d:10:s') #pin 10 Arduino

# Callback function to get mouse click coordinates
def click_event(event, x, y, flags, params):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pressed", x, y)
        point = (x, y)
    if event == cv2.EVENT_RBUTTONDOWN:
        save_image(frame)

# Function to write frame to video
def write_video(video_writer, frame):
    video_writer.write(frame)

count = 0
# Function to save image
def save_image(frame):
    global count
    count+=1
    cv2.imwrite(f'gallery/test{count}.png', frame)

# Create a window and set the mouse callback function
cv2.namedWindow("YOLOv8 Tracking")
cv2.setMouseCallback("YOLOv8 Tracking", click_event)

# Open the video file
cap = cv2.VideoCapture(0)

# Video writer setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# new_width, new_height = 1920, 1080
size = (frame_width, frame_height)
vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP42'), 25.0, size)

# YOLO tracking parameters
color = (0, 255, 0)
line_width = 2
radius = 7
point = (320, 240)
blink_interval = 500 
servoPos = [90, 90] # initial servo position

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    k = cv2.waitKey(1)

    if success:
        if k & 0xFF == ord("q"):
            break
        elif k & 0xFF == ord("r"):
            capture_video = not capture_video  # Toggle video recording on 'r' key
        elif k & 0xFF == ord("c"):
            save_image(frame)  # Save an image when the 'c' key is pressed

        if capture_video:
            write_video(vid_writer, frame)
            current_time = int(time.time() * 1000)
            show_text = current_time % (2 * blink_interval) < blink_interval
            if show_text:
                cv2.putText(frame, "Recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.6, iou=0.5, max_det=2, classes = [0])

        # Initialize variables
        detected_objects = []
        closest_object_index = None
        min_distance = float('inf')
        min_obj_center = (0,0)

        # Extract bounding boxes from results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detected_objects.append((x1, y1, x2, y2))

        # Calculate minimum distance
        for i, (x1, y1, x2, y2) in enumerate(detected_objects):
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = ((object_center[0] - point[0]) ** 2 + (object_center[1] - point[1]) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_object_index = i
                a, b, c, d = detected_objects[i]
                min_obj_center = ((a + c) // 2, (b + d) // 2)
                
        
        fx = min_obj_center[0]
        fy = min_obj_center[1]

        servoX = np.interp(fx, [0, 640], [0, 180])   
        servoY = np.interp(fy, [0, 480], [0, 180])
        print(servoX,servoY)

        if servoX < 0:
            servoX = 0
        elif servoX > 180:
            servoX = 180
        if servoY < 0:
            servoY = 0
        elif servoY > 180:
            servoY = 180

        servoPos[0] = servoX
        servoPos[1] = servoY

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.circle(annotated_frame, point, radius, color, line_width)
        point = min_obj_center
        # cv2.circle(annotated_frame, min_obj_center if object_center != None else (0, 0), radius, color, line_width)

        # board.digital[pin_10].write(servoPos[0])
        # board.digital[pin_9].write(servoPos[1])
        # servo_pinX.write(servoPos[0])
        # servo_pinY.write(servoPos[1])
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
# vid_writer.release()
cv2.destroyAllWindows()
