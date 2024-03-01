import cv2

def cv_main():
    cascade_path = './opencv-4.8.0/data/haarcascades/'

    camera_opencv(cascade_path)

def camera_opencv(cascade_path):

    cascade_path_1 = cascade_path + '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path_1)

    # ID 0でデバイスを開きます
    for camera_number in range(0, 10):
        cap = cv2.VideoCapture(camera_number)
        ret, frame = cap.read()

        if ret is True:
            break
        else:
            cap.release()


    while True:
        ret, img = cap.read()

        if ret is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = img[y: y + h, x: x + w]
                face_gray = gray[y: y + h, x: x + w]

            cv2.putText(img, 'Face Detect', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,200), 2, cv2.LINE_AA)
            cv2.imshow('video image', img)

        key = cv2.waitKey(10)
        #エスケープキー
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    cv_main()