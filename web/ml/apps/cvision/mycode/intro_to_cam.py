import cv2

dir_images_output = '../data/images/output/'
file_name_output = 'out.avi'
file_name_output = dir_images_output + file_name_output

cap = cv2.VideoCapture(0)  # 0 default camera; 1 or 2 or 3 for different cam
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(file_name_output, fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret -- True:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(w, h)
        out.write(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()


