import numpy as np
import cv2

dir_images_output = '../data/images/output/'
dir_images = '../data/images/'
file_name = 'lena.jpg'
file_name = dir_images + file_name
file_name_output = 'lena_copy.png'
file_name_output = dir_images_output + file_name_output

# img = cv2.imread(file_name, cv2.IMREAD_COLOR)  # cv2.IMREAD_UNCHANGED) # cv2.IMREAD_GRAYSCALE) = 0 # cv2.IMREAD_COLOR) 1
# https://www.google.com/search?q=rgb+color+picker&oq=rgb+color+picker&aqs=chrome..69i57j0l7.6546j0j7&sourceid=chrome&ie=UTF-8
img = 100*np.ones([512, 512, 3], np.uint8)

img = cv2.line(img, (0, 0), (255, 255), (173, 33, 23), 5)
img = cv2.arrowedLine(img, (0, 255), (255, 255), (255, 0, 0), 5)
img = cv2.rectangle(img, (385, 255), (510, 128), (0, 0, 255), -1)  # 10)
img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
font = cv2.FONT_HERSHEY_COMPLEX
img = cv2.putText(img, 'OpenCv', (10, 500), font, 4, (255, 255, 255), 10, cv2.LINE_AA)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()