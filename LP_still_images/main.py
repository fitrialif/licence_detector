import cv2
import numpy as np
import os
import time
from readplate import readplate

def Rect(img, rect,box):
    # Let cnt be the contour and img be the input

# Assign width and height of image to be crop
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]

    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90
    center = rect[0]

    size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    Rotatedrect = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
    return Rotatedrect

def img_processing(img):

    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    str_element_size=(16,4)
    kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    enhanced_gray=cv2.filter2D(imgGrayscale, -1, kernel)
    imgBlurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)# blur
    equal_histogram = cv2.equalizeHist(imgBlurred)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, str_element_size)
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel2,iterations=10)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    h,sobel = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
    images = [img, imgGrayscale, imgBlurred, equal_histogram, sub_morp_image, sobel]
    
    #cv2.namedWindow("Gray",cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Blurred",cv2.WINDOW_NORMAL)
    #cv2.namedWindow("submorph",cv2.WINDOW_NORMAL)
    #cv2.namedWindow("threshold",cv2.WINDOW_NORMAL)
    
    #cv2.imshow("Gray", imgGrayscale)
    #cv2.imshow("Blurred", imgBlurred)
    #cv2.imshow("submorph", sub_morp_image)
    #cv2.imshow("threshold", sobel)
   
    return sobel, imgBlurred

def density(possible_lic,w,h,x,y):
    white_pix=0
    
    plate_possible = cv2.getRectSubPix(possible_lic, (int(w),int(h)), (x, y))		
    z,plate_possible = cv2.threshold(plate_possible,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    for X in range(plate_possible.shape[0]):
		for Y in range(plate_possible.shape[1]):
			if plate_possible[X][Y] == 255:
			    white_pix += 1
    edge_density = float(white_pix)/(plate_possible.shape[0]*plate_possible.shape[1])
    return edge_density

def LPdata():
    arr = []
    inp = open ("Licenseplates.txt","r")
    for line in inp.readlines():
        # add a new sublist
        arr.append([])
        # loop over the elemets, split by whitespace
        for i in line.split():
            # convert to integer and append to the last
            # element of the list
            arr[-1].append(str(i))
    return arr

def main():
    t1 = time.time()                                        # Take the time when program starts
    Lp = LPdata()
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread("1.jpg")                            # open image

    if img is None:                                         # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)

    morphed_sobel, Blurred=img_processing(img)
   
    # Finding contours in image
    _,contours,_=cv2.findContours(morphed_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   
    w,h,x,y = 0,0,0,0
    
    # Taking all countours that were found and looking for text   
    for contour in contours:
        area = cv2.contourArea(contour)             # Calculating the area of each contour
        
        rect = cv2.minAreaRect(contour)             
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # If contour area is within set boundaries it cuts the countour out of the image 
        if area > 700 and area < 8000:
            [x,y,w,h] = cv2.boundingRect(contour)       # find x,y coordinates of the corners of the contour
            if w>4*h and w<6*h:                         # Check if contour is within set aspect ratio of a numberplate
                crop_img = Blurred[y:y+h,x:x+w]         # Crop contour out of the image
                edgedensity=density(crop_img,w,h,x,y)   # Find edge density of the contour
                if edgedensity>0.5:
                    plate = Rect(img,rect,box)          # Correct the angle of the contour so it is straight
                    cv2.drawContours(img,[box],0,(255,0,0),2)       # Draw contour on the original image
                    license = readplate(plate)                      # Find licence plate number in the contour
                    print "Number plate found: " + license          # Print the licence plate number if found
                    
                    for text in Lp:
                        if str(license) in text:
                            SGSdata=text
                            print SGSdata
                            cv2.putText(img,str(license),(x,y+h+60), font, 2,(0,0,255),3,cv2.LINE_AA)
                            cv2.putText(img,str(SGSdata[1] +" "+ SGSdata[3]),(x,y+h+100), font, 1,(0,0,255),2,cv2.LINE_AA)
                            print "Next inspection: " + SGSdata[5]

                    

    
    ttime=(time.time()-t1)*1000                     # Calculate how long the program took to process the image
    print 'Time taken: %d ms'%(ttime)               # Print how long it took to process the image

    cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
    cv2.imshow("Original", img)                         # Display original image with found contour
    cv2.waitKey()                                   # Wait for user input
    cv2.destroyAllWindows()                         # Close all windows
    return

if __name__ == "__main__":

    main()
