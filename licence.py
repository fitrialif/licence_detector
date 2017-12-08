import cv2

import numpy as np
import argparse
import os
from copy import deepcopy, copy
import random
from matplotlib import pyplot as plt
###################################################################################################

def main():

    imgOriginal = cv2.imread("16.jpg")               # open image



    if imgOriginal is None:                             # if image was not read successfully

        print "error: image not read from file \n\n"        # print error message to std out

        os.system("pause")                                  # pause so user can see error message

        return                                              # and exit function (which exits program)

    img=imgOriginal.copy()
    image=imgOriginal.copy()
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", help = "path to the image file")
    #ap.add_argument("-r", "--radius", type = int,
	#help = "radius of Gaussian blur; must be odd")
    #args = vars(ap.parse_args())
    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)        # convert to grayscale

    #imaa=cv2.cvtColor(imgOriginal,cv2.COLOR_RGB2HSV)
    roi=[]
    imgBlurred = cv2.GaussianBlur(imgGrayscale, (5, 5), 0)              # blur
    #520mmx110
 
    #intRandomBlue = random.randint(0, 255)

    
    #plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    #plt.title('Original'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    #plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    #plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    #plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    #plt.show()



    #plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
    #plt.title('Original'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
    #plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
    #plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    #plt.show
    #intRandomRed = random.randint(0, 255)

    ar=520/110
    gray_image = deepcopy(imgGrayscale)

    gray_image = cv2.medianBlur(gray_image, 3)
    
    
    
    #laplacian = cv2.Laplacian(gray_image,cv2.CV_64F)
    #sobelx = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
    #sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)
    ##intRandomGreen = random.randint(0, 255)
    #sobelx8u = cv2.Sobel(gray_image,cv2.CV_8U,1,0,ksize=5)
    #cv2.imshow('original',imgOriginal)
    #cv2.imshow('lapl',laplacian)
    #cv2.imshow('sobelx',sobelx)
    #cv2.imshow('sobely',sobely)
    #cv2.imshow('sobelx8u',sobelx8u)


    #    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    #sobelx64f = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
    #abs_sobel64f = np.absolute(sobelx64f)
    #sobel_8u = np.uint8(abs_sobel64f)
    #cv2.imshow('sobelx64',sobelx64f)
    #cv2.imshow('sobely',abs_sobel64f)
    #cv2.imshow('sobel8u',sobel_8u)

    #imgCanny = cv2.Canny(imgBlurred, 200, 255) 
    #cv2.imshow('Original',imgOriginal)
    #gray_image=imgCanny
    gray_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55,5)

    _,contours,_ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('thresh',gray_image)
    #cv2.drawContours(imgOriginal, contours, -1,(0,0,255),6)
    #cv2.imshow('conssssssours',imgOriginal)
    w,h,x,y = 0,0,0,0
    #boundaries = [([71, 40, 31], [50, 36, 21]),([60, 36, 24], [28, 16, 10]),([25, 146, 190], [62, 174, 250]),([103, 86, 65], [145, 133, 128])]
    
    #for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	    #lower = np.array(lower, dtype = "uint8")
	    #upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	    #mask = cv2.inRange(image, lower, upper)
	    #output = cv2.bitwise_and(image, image, mask = mask)
 
	# show the images
    #cv2.imshow("images", np.hstack([imgOriginal, gray_image]))
    for contour in contours:
        #intRandomBlue = random.randint(0, 255)
        #intRandomGreen = random.randint(0, 255)
        #intRandomRed = random.randint(0, 255)
        area = cv2.contourArea(contour)
        #[x,y,w,h] = cv2.boundingRect(contour)
        #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
       

			# rough range of areas of a license plate

        if area > 400 and area < 10000:
            [x,y,w,h] = cv2.boundingRect(contour)
            #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
            #cv2.imshow('nabbi',imgOriginal)
            #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
            #cv2.drawContours(imgOriginal, contours, -1,(0,0,255),6)
            #cv2.imshow('nabbi',imgOriginal)
			# rough dimensions of a license plate
            

            if w>4*h and w<6*h:#w > 40 and w < 900 and h > 50 and h < 600:
                roi.append([x,y,w,h])
                cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
                #cv2.imshow('possible plates',imgOriginal)
              #  print area
                print area
                
                #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
                #print w
                #cv2.imshow('Boxes',imgOriginal)



    if len(roi) > 1:

		[x,y,w,h] = roi[0]

		plate_image = imgOriginal[y:y+h,x:x+w]

		#plate_image_char = deepcopy(plate_image)
		#cv2.imshow('coplaters',plate_image)
        #else:
         #   continue
    
    titles = ['Original Image', 'Adaptive Gaussian Thresholding',
            'Contours', 'Plate image']
    images = [image, gray_image, imgOriginal, plate_image]

    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show() 
        #####gamla
    #imgCanny = cv2.Canny(imgBlurred, 200, 250)                          # get Canny edges
    #ret, threshold = cv2.threshold(imgBlurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #_, contours,_ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #print len(contours)

    #cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)        # create windows, use WINDOW_AUTOSIZE for a fixed window size

    #cv2.namedWindow("imgCanny", cv2.WINDOW_AUTOSIZE)           # or use WINDOW_NORMAL to allow window resizing
    #cv2.drawContours(imgOriginal, contours, 1,(0,0,255),6)

    #gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imgBlurred)
    #image = imgOriginal.copy()
    #cv2.circle(image, maxLoc, 10, (255, 0, 0), 2)


    #args["radius"]
    #cv2.imshow("imsnal", imaa) 
    #cv2.imshow("imgOriginal", imgOriginal)         # show windows
    #cv2.imshow("imgOriginal", imgOriginal) 
    #cv2.imshow("imgCanny", imgCanny)



    cv2.waitKey()                               # hold windows open until user presses a key



    cv2.destroyAllWindows()                     # remove windows from memory



    return



###################################################################################################

if __name__ == "__main__":

    main()
