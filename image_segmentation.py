import cv2 as cv
import numpy as np

img = cv.imread('enter the image path here')
#changing the image into HSV for better color segmentation
hsvimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Creating a mask for blue, to seperate the ocean using the upper and lower boundary values for the color in hsv
lower_blue = np.array([95, 80, 50]) 
upper_blue = np.array([130, 255, 255])
blue_mask = cv.inRange(hsvimg, lower_blue, upper_blue)

#Doing the same for green
lower_green = np.array([50, 50, 50])
upper_green = np.array([65, 255, 255])
green_mask = cv.inRange(hsvimg, lower_green, upper_green)

#Creating a blank image of same dimensions as original to "paste" the masks on
maskedimg = np.zeros(img.shape, dtype='uint8')

#Using the blue_mask to get the map of the blue areas and then assigning them the plain blue on the blank image
maskedimg[blue_mask>0]=[255,0,0]
#Doing the same for the green areas
maskedimg[green_mask>0]=[0,255,255]

#In the final image, the land and water areas are clearly segmented with plain yellow for land and plain blue for water
cv.imshow('Output', maskedimg)
cv.waitKey(0)
cv.destroyAllWindows()
#in the sample images given , one rescue pad was light blue in colour which was also identified as the ocean body.I tried varying the hsv values but nothing changed
