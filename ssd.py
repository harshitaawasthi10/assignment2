import cv2
import tensorflow as tf

model = tf.saved_model.load('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz')

video_path = r"C:\Users\LENOVO\Desktop\WhatsApp Video 2024-12-21 at 21.44.13_342a5fef.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of video frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of video frames

out = cv2.VideoWriter(
    'output_ssd.avi',  # Output file name
    cv2.VideoWriter_fourcc(*'XVID'),  # Codec for output video
    fps,  # Frames per second
    (width, height)  # Frame size
)

def preprocess(frame):
    # Resize and normalize frame for SSD input
    input_tensor = tf.image.resize_with_pad(frame, 300, 300)
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
    return tf.expand_dims(input_tensor, 0)

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), break the loop
        break

    # Perform object detection
    input_tensor = preprocess(frame)
    detections = model(input_tensor)  # Model output

    # Draw bounding boxes on the frame
    detection_boxes = detections['detection_boxes'][0].numpy()  # Extract boxes
    detection_classes = detections['detection_classes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()

    for i, box in enumerate(detection_boxes):
        if detection_scores[i] > 0.5:  # Confidence threshold
            h, w, _ = frame.shape
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * w), int(ymin * h))
            end_point = (int(xmax * w), int(ymax * h))
            frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            label = str(detection_classes[i])
            frame = cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Write the annotated frame to the output video
    out.write(frame)

    cv2.imshow('SSD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
