import cv2
import numpy as np
import os
from copy import deepcopy, copy
import random
from matplotlib import pyplot as plt
import time
from PIL import Image
from pytesseract import image_to_string
import unicodedata
from readplate import readplate
###################################################################################################
def img_processing(img):
    #t2 = time.time()
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    str_element_size=(16,4)
    kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    enhanced_gray=cv2.filter2D(imgGrayscale, -1, kernel)
    imgBlurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)# blur
    equal_histogram = cv2.equalizeHist(imgBlurred)

    #t6 = time.time()
    #sobelx = cv2.Sobel(imgBlurred, -1, 0, 1)
    #sobelx64f = cv2.Sobel(equal_histogram,-1,0,1,ksize=5)
    #abs_sobel64f = np.absolute(sobelx64f)
    #sobelx = np.uint8(abs_sobel64f)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #sobelx = cv2.Sobel(equal_histogram, -1, 0,1) #Sobel of the equalized hist grayscale image for enhanced edges
    #cv2.imshow("sobel222", sobelx)
    #cv2.imshow("sobelasss", abs_sobelx64f)
    #t7 = time.time()
    se = cv2.getStructuringElement(cv2.MORPH_RECT, str_element_size)
    #t8 = time.time()
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel2,iterations=10)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    h,sobel = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gray",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Blurred",cv2.WINDOW_NORMAL)
    cv2.namedWindow("submorph",cv2.WINDOW_NORMAL)
    cv2.namedWindow("threshold",cv2.WINDOW_NORMAL)
    

    images = [img, imgGrayscale, imgBlurred, equal_histogram, sub_morp_image, sobel]
    #cv2.imshow("Original", img)
    cv2.imshow("Gray", imgGrayscale)
    cv2.imshow("Blurred", imgBlurred)
    cv2.imshow("submorph", sub_morp_image)
    cv2.imshow("threshold", sobel)
    #sobel = cv2.Canny(sobel,200,255)
    #sobel= cv2.adaptiveThreshold(sobelx, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43,2)
    #t9 = time.time()
    #gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)
    
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

def main():
    
    e1 = cv2.getTickCount()
    
    imgOriginal = cv2.imread("CV/3.jpg")               # open image

    if imgOriginal is None:                             # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    t1 = time.time()
    print t1
    img=np.copy(imgOriginal)
    roi=[]
    morphed_sobel, Blurred=img_processing(img)
    cv2.imshow("sobel", morphed_sobel)
    t333=(time.time()-t1)*1000
    print t333
    #ar=520/110
    
    cv2.namedWindow("Orig",cv2.WINDOW_NORMAL)

    #gray_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55,5)

#    _,contours,_ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _,contours,_=cv2.findContours(morphed_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    w,h,x,y = 0,0,0,0

    for contour in contours:

        #rectangle=cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)
        #[x,y,w,h] = cv2.boundingRect(contour)
        #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)


        if area > 700 and area < 8000:
            [x,y,w,h] = cv2.boundingRect(contour)
            #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
            #cv2.imshow('nabbi',imgOriginal)
            #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
            #cv2.drawContours(imgOriginal, contours, -1,(0,0,255),6)
            #cv2.imshow('nabbi',imgOriginal)
			# rough dimensions of a license plate
            

            if w>4*h and w<6*h:#w > 40 and w < 900 and h > 50 and h < 600:   ##Aspect ratio of the licence plates
                ar=w/h
                
                #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
                #possibleplate=density()
                crop_img = Blurred[y:y+h,x:x+w]
                edgedensity=density(crop_img,w,h,x,y)
                #print edgedensity
                if edgedensity>0.5:
                    roi.append([x,y,w,h])
                    img_new = Image.fromarray(crop_img)
                    plate = image_to_string(img_new, lang='eng')#,config="-c tessedit_char_whitelist=ABCDEFGHJKLMNPQRSTVXYZ1234567890 -psm 10") #image_to_string(img_new, lang='eng')
                    plate_new = unicodedata.normalize('NFKD', plate).encode('ascii','ignore')
                    cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,255), 6)

                    cropped=img[y:y+h,x:x+w]
                    cv2.imshow("cropped", cropped)

                    #cv2.imshow("cropped", crop_img)

                    #print ar
                   # if plate_new: #plate_new:
                    #    print ar
                        #roi.append([x,y,w,h])
                        
                        
                     #   print plate_new #plate_new
                      #  print 'ok' 
                #crop_or=np.copy(crop_img)
                
                
                #cv2.imshow("Orig", imgOriginal)
                #crop_img=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                #b,crop_img = cv2.threshold(crop_img,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                #crop_img=cv2.GaussianBlur(crop_img, (3,3), 0)
                #readers=readplate(crop_or,crop_img)
                #print readers    
                


                #cv2.imshow('lic',crop_img)
                #plate = image_to_string(img_new, lang='eng')
                
                #plate_new = unicodedata.normalize('NFKD', plate).encode('ascii','ignore')
            
                #if plate_new:
                 #   roi.append([x,y,w,h])
                  #  cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,255), 6)
                   # cv2.imshow("cropped", crop_img)
                    #print plate_new
                    #print 'ok' 
                #cv2.imshow('Contours',imgOriginal)
                #print area
                #box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
                #box = np.int0(box)
                #cv2.drawContours(im,[box],0,(0,0,255),2)
                
                #cv2.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 6)
                #print w
                #cv2.imshow('Boxes',imgOriginal)

    cv2.imshow("Orig", imgOriginal)
    ttime=(time.time()-t1)*1000
    print 'Time taken: %d ms'%(ttime)
    
    if len(roi) >= 1:

        [x,y,w,h] = roi[0] #plate_image = imgOriginal[y:y+h,x:x+w]


    else:   
        plate_image=img = np.zeros((300,512,3), np.uint8)
    
    #titles = ['Original Image', 'Adaptive Gaussian Thresholding',
    #        'Contours', 'Plate image']
    #images = [morphed_sobel, img, imgOriginal, crop_img]

    #for i in xrange(4):
    #    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #    plt.title(titles[i])
    #    plt.xticks([]),plt.yticks([])
    #plt.show() 
        #####gamla
    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print( t )
    cv2.waitKey()                               # hold windows open until user presses a key


    cv2.destroyAllWindows()                     # remove windows from memory


    return



###################################################################################################

if __name__ == "__main__":

    main()

