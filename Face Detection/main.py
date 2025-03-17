import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Access the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not access the camera")
    exit()

while True:
    # Capture frame-by-frame from camera
    ret, frame = cap.read()

    # If frame was not captured correctly, skip this iteration
    if not ret:
        print("Error: Couldn't retrieve frame")
        continue

    # Convert the frame to grayscale for face detection and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(300,300))

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 0, 0), 2)

    #Display the resulting frame with detected face
    cv2.imshow('Face Detection', frame)

    # Break the loop if the user press the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break