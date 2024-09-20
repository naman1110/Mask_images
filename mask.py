import cv2
import numpy as np

# Load the image
image = cv2.imread('beer.jpg')

# Convert the image to HSV color space for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the threshold for white color in HSV space
# These values may need adjustment based on your specific image
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 20, 255])

# Create a mask for the white background
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Invert the mask to get the car (non-white areas)
mask_car = cv2.bitwise_not(mask_white)

# Optionally, perform morphological operations to remove noise
kernel = np.ones((3, 3), np.uint8)
mask_car = cv2.morphologyEx(mask_car, cv2.MORPH_OPEN, kernel)
mask_car = cv2.morphologyEx(mask_car, cv2.MORPH_DILATE, kernel)

# Invert the mask so that the car is black and background is white
car_mask_bw = cv2.bitwise_not(mask_car)

# Save the black-and-white mask image
cv2.imwrite('campus_mask1.png', car_mask_bw)

# Display the result (optional)
cv2.imshow('campus Mask Black and White', car_mask_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()