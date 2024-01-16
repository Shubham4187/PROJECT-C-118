import cv2

# Load the pre-trained body classifier
body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Open a video capture object (you can replace 'walking.avi' with 0 for webcam input)
cap = cv2.VideoCapture('walking.avi')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there are no more frames

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Draw rectangles around detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Body Detection', frame)

    # Break the loop if 'Space' key is pressed
    if cv2.waitKey(1) == 32:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
