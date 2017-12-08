#import sys
import glob
import math
import time
import cv2
# Importing the Opencv Library
import numpy as np

# Importing NumPy,which is the fundamental package for scientific computing with Python
def validate_contour(contour, img, aspect_ratio_range, area_range):
    rect = cv2.minAreaRect(contour)
    img_width = img.shape[1]
    img_height = img.shape[0]
    box = cv2.boxPoints(rect) 
    box = np.int0(box)

    X = rect[0][0]
    Y = rect[0][1]
    angle = rect[2] 
    width = rect[1][0]
    height = rect[1][1]

    angle = (angle + 180) if width < height else (angle + 90)

    output=False
    
    if (width > 0 and height > 0) and ((width < img_width/2.0) and (height < img_width/2.0)):
    	aspect_ratio = float(width)/height if width > height else float(height)/width
        if (aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]):
        	if((height*width > area_range[0]) and (height*width < area_range[1])):

        		box_copy = list(box)
        		point = box_copy[0]
        		del(box_copy[0])
        		dists = [((p[0]-point[0])**2 + (p[1]-point[1])**2) for p in box_copy]
        		sorted_dists = sorted(dists)
        		opposite_point = box_copy[dists.index(sorted_dists[1])]
        		tmp_angle = 90

        		if abs(point[0]-opposite_point[0]) > 0:
        			tmp_angle = abs(float(point[1]-opposite_point[1]))/abs(point[0]-opposite_point[0])
        			tmp_angle = rad_to_deg(math.atan(tmp_angle))

        		if tmp_angle <= 45:
        			output = True
    return output

def deg_to_rad(angle):
	return angle*np.pi/180.0

def rad_to_deg(angle):
	return angle*180/np.pi
def enhance(img):
	kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
	return cv2.filter2D(img, -1, kernel)
# Reading Image
t1 = time.time()
img = cv2.imread("26.jpg")
raw_image = np.copy(img)
input_image = np.copy(raw_image)
#cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Original Image",img)
# Display image

# RGB to Gray scale conversion
#img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Gray Converted Image",img_gray)
# Display Image

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
#noise_removal = cv2.bilateralFilter(img_gray,9,75,75)#cv2.GaussianBlur(img_gray, (5, 5), 0)
#cv2.namedWindow("Noise Removed Image",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Noise Removed Image",noise_removal)
# Display Image

# Histogram equalisation for better results
#equal_histogram = cv2.equalizeHist(noise_removal)
#cv2.namedWindow("After Histogram equalisation",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("After Histogram equalisation",equal_histogram)
# Display Image
se_shape=(16,4)
#raw_image = cv2.imread(name,1)
#input_image = np.copy(raw_image)

gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
gray = enhance(gray)
gray = cv2.GaussianBlur(gray, (5,5), 0)
gray = cv2.Sobel(gray, -1, 1, 0)
h,sobel = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
se = cv2.getStructuringElement(cv2.MORPH_RECT, se_shape)
gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)
ed_img = np.copy(gray)
cv2.imshow("morph",gray)

_,contours,_=cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

font = cv2.FONT_HERSHEY_SIMPLEX


for contour in contours:
	aspect_ratio_range = (2.2, 12)
	area_range = (500, 18000)

	#if options.get('type') == 'rect':
	#	aspect_ratio_range = (2.2, 12)
	#	area_range = (500, 18000)

	#elif options.get('type') == 'square':
	#	aspect_ratio_range = (1, 2)
	#	area_range = (300, 8000)

	if validate_contour(contour, gray, aspect_ratio_range, area_range):
		rect = cv2.minAreaRect(contour)  
		box = cv2.boxPoints(rect) 
		box = np.int0(box)  
		Xs = [i[0] for i in box]
		Ys = [i[1] for i in box]
		x1 = min(Xs)
		x2 = max(Xs)
		y1 = min(Ys)
		y2 = max(Ys)

		angle = rect[2]
		if angle < -45:
			angle += 90 

		W = rect[1][0]
		H = rect[1][1]
		aspect_ratio = float(W)/H if W > H else float(H)/W

		center = ((x1+x2)/2,(y1+y2)/2)
		size = (x2-x1, y2-y1)
		M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0);
		tmp = cv2.getRectSubPix(ed_img, size, center)
		tmp = cv2.warpAffine(tmp, M, size)
		TmpW = H if H > W else W
		TmpH = H if H < W else W
		tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))
		__,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		white_pixels = 0

		for x in range(tmp.shape[0]):
			for y in range(tmp.shape[1]):
				if tmp[x][y] == 255:
					white_pixels += 1

		edge_density = float(white_pixels)/(tmp.shape[0]*tmp.shape[1])

		tmp = cv2.getRectSubPix(raw_image, size, center)
		tmp = cv2.warpAffine(tmp, M, size)
		TmpW = H if H > W else W
		TmpH = H if H < W else W
		tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))

		if edge_density > 0.5:
			cv2.drawContours(input_image, [box], 0, (127,0,255),2)
ttime=(time.time()-t1)*1000
print 'Time taken: %d ms'%(ttime)
cv2.imshow("cont",input_image)
# Morphological opening with a rectangular structure element
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=10)
#cv2.namedWindow("Morphological opening",cv2.WINDOW_NORMAL)
# Creating a Named window to display image

#cv2.imshow("Morphological opening",morph_image)
# Display Image

# Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
#sub_morp_image = cv2.subtract(equal_histogram,morph_image)
#cv2.namedWindow("Subtraction image", cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Subtraction image", sub_morp_image)
# Display Image

# Thresholding the image
#ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
#cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Image after Thresholding",thresh_image)
# Display Image

# Applying Canny Edge detection
#canny_image = cv2.Canny(thresh_image,200,255)
#cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Image after applying Canny",canny_image)
# Display Image
#canny_image = cv2.convertScaleAbs(canny_image)

# dilation to strengthen the edges
#kernel = np.ones((3,3), np.uint8)
# Creating the kernel for dilation
#dilated_image = cv2.dilate(canny_image,kernel,iterations=2)
#cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
# Creating a Named window to display image
#cv2.imshow("Dilation", dilated_image)
# Displaying Image

# Finding Contours in the image based on edges
#new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
#cv2.drawContours(img, contours, -1,(0,0,255),6)
#cv2.imshow("Image with  Contoursall",img)
# Sort the contours based on area ,so that the number plate will be in top 10 contours
#screenCnt = None
# loop over our contours
#for c in contours:
# # approximate the contour
# peri = cv2.arcLength(c, True)
# approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
# # if our approximated contour has four points, then
# # we can assume that we have found our screen
# if len(approx) == 4:  # Select the contour with 4 corners
#  screenCnt = approx
#  break
#final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
## Drawing the selected contour on the original image
#cv2.namedWindow("Image with Selected Contour",cv2.WINDOW_NORMAL)
## Creating a Named window to display image
#cv2.imshow("Image with Selected Contour",final)

## Masking the part other than the number plate
#mask = np.zeros(img_gray.shape,np.uint8)
#new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
#new_image = cv2.bitwise_and(img,img,mask=mask)
#cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
#cv2.imshow("Final_image",new_image)

## Histogram equal for enhancing the number plate for further processing
#y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))
## Converting the image to YCrCb model and splitting the 3 channels
#y = cv2.equalizeHist(y)
## Applying histogram equalisation
#final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)
## Merging the 3 channels
#cv2.namedWindow("Enhanced Number Plate",cv2.WINDOW_NORMAL)
## Creating a Named window to display image
#cv2.imshow("Enhanced Number Plate",final_image)
# Display image
cv2.waitKey() # Wait for a keystroke from the user
