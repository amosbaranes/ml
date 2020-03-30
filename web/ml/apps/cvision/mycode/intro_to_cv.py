import cv2

dir_images_output = '../data/images/output/'
dir_images = '../data/images/'
file_name = 'lena.jpg'
file_name = dir_images + file_name
file_name_output = 'lena_copy.png'
file_name_output = dir_images_output + file_name_output

img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_UNCHANGED) # cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR)

print(img)
cv2.imshow('image', img)
k = cv2.waitKey(0)  # 5000)

if k == 27:     # escap key
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite(file_name_output, img)
    cv2.destroyAllWindows()


