import cv2
# for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #to load the xml files
# resolution of the webcam
screen_width = 1280       
screen_height = 720
# default webcam
stream = cv2.VideoCapture(0)
while(True):
    # capture frame-by-frame
    (ret, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        color = (0, 255, 255) # yellow in BGR
        stroke = 2 #thickness
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
    # show the frame
    cv2.imshow("facedetection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out
     break                  # of the loop
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
