import cv2
import numpy as np
# import random
# img = cv2.imread('logo.jpg', -1)
# print(img)
# print(img[50][400])
# color: (blue, green, red)

#1
# for i in range(100):
#     for j in range(img.shape[1]):
#         img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# cv2.imwrite('new_img.jpg', img)
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#2
#copy part of the image and then paste
# tag = img[350:400, 200:250]  #row 500 to 700, 600 to 900
# img[100:150, 50:100] = tag
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Mirror video multiple times
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read() #ret will tell us if the camera is working properly
#     width = int(cap.get(3)) #3 -> the property of capture, width
#     height = int(cap.get(4))
#     image = np.zeros(frame.shape, np.uint8)
#     smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    #1/4 of the original image
#     image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)  
#     image[height//2:, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
#     image[:height//2, width//2:] = smaller_frame
#     image[height//2:, width//2:] = smaller_frame

#     cv2.imshow('frame', image)
#     if cv2.waitKey(1) == ord('q'):  #return the int we hit the key
#         break
# cap.release()
# cv2.destroyAllWindows()

#Drawing
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read() #ret will tell us if the camera is working properly
#     width = int(cap.get(3)) #3 -> the property of capture, width
#     height = int(cap.get(4))

#     img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 10)
#     img = cv2.line(img, (0, height), (width, 0), (0, 255, 0), 5)

#     img = cv2.rectangle(img, (100, 100), (200, 200), (128, 128, 128), 4)    #if the line thickness is -1, it will fill the entire shape
#     img = cv2.circle(img, (300, 300), 60, (128, 128, 128), 4)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     img = cv2.putText(img, 'YJ is Great!', (200, height - 10), font, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

#     cv2.imshow('frame', img)
#     if cv2.waitKey(1) == ord('q'):  #return the int we hit the key
#         break
# cap.release()
# cv2.destroyAllWindows()

#Colors and Color Detection

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read() #ret will tell us if the camera is working properly
#     width = int(cap.get(3)) #3 -> the property of capture, width
#     height = int(cap.get(4))

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([90, 50, 50])
#     upper_blue = np.array([130, 255, 255])

#     #if we show the mask itself, it will be a black and white picture     
#     #within the range pixel is 1;if not, 0   
#     mask = cv2.inRange(hsv, lower_blue, upper_blue) 

#     result = cv2.bitwise_and(frame, frame, mask=mask)   #if the pixel is blue, keep it; otherwise, make it black

#     cv2.imshow('frame', result)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#Corner Detection
#find all the corners in gray, then draw circles on colored img

# img = cv2.imread('chessboard.jpg')
# img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale

# corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
# corners = np.int0(corners)  #convert from float to int
# for corner in corners:
#     x, y = corner.ravel()   #flatten the array, e.g. [[1, 2], [2, 1]] -> [1, 2, 2, 1]
#     cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

# #draw random lines between corner1 and corner2
# for i in range(len(corners)):
#     for j in range(i+1, len(corners)):
#         corner1 = tuple(corners[i][0])  #need to be flattened for corners[i] here, so we add[0]
#         corner2 = tuple(corners[j][0])
#         color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))   #conver 64 int to 8 int using map, and then convert the list to tuple
#         cv2.line(img, corner1, corner2, (color), 1)



# cv2.imshow('Frame', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Template Matching
#The size of the template image should be close to the object size in the base image
# img = cv2.resize(cv2.imread('soccer_practice.jpg', 0), (0, 0), fx=0.5, fy=0.5)  #load the gray scale required by the algorithm
# template = cv2.imread('ball.PNG', 0)
# template = cv2.resize(cv2.imread('shoe.PNG', 0), (0, 0), fx=0.5, fy=0.5)
# h, w = template.shape
#the dimension for the grey scale image is 2d, no need channel
# print(img)
#There are 6 methods in the template matching, you can try one by one
# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
#loop through all the methods
# for method in methods:
#     img2 = img.copy()
#     result = cv2.matchTemplate(img2, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     # print(min_loc, max_loc)
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         location = min_loc
#     else:
#         location = max_loc

#     bottom_right = (location[0] + w, location[1] + h)
#     cv2.rectangle(img2, location, bottom_right, 255, 5)  #255 means black
#     cv2.imshow('Match', img2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#Face and Eye Detection
cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('C:\Python\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Python\haarcascade_eye.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    

