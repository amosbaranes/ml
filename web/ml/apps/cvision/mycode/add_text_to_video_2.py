import cv2, datetime

cap = cv2.VideoCapture(0)  # 0 default camera; 1 or 2 or 3 for different cam

# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3000)  # https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if ret -- True:
        font = cv2.FONT_HERSHEY_COMPLEX
        # text = 'Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
        date_time = str(datetime.datetime.now())
        frame = cv2.putText(frame, date_time, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


