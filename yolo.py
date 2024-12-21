import cv2  
import torch  

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video_path = r"C:\Users\LENOVO\Desktop\WhatsApp Video 2024-12-21 at 21.44.13_342a5fef.mp4" # Replace with your video file path
cap = cv2.VideoCapture(video_path)  # Open the video file

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()  # Exit the program if the video cannot be opened

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of video frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of video frames

out = cv2.VideoWriter(
    'output_yolo.avi',  # Output file name
    cv2.VideoWriter_fourcc(*'XVID'),  # Codec for output video
    fps,  # Frames per second
    (width, height)  # Frame size
)
# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), break the loop
        break

    results = model(frame)

    result_frame = results.render()[0]  # Annotated frame

    cv2.imshow('YOLOv5', result_frame)

    out.write(result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  
out.release()  
cv2.destroyAllWindows()  