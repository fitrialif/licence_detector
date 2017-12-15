#import numpy as np
#import cv2
#cap = cv2.VideoCapture('Ingi.MOV')
#fgbg = cv2.createBackgroundSubtractorMOG()
#while(1):
#    ret, frame = cap.read()
#    fgmask = fgbg.apply(frame)
#    cv2.imshow('frame',fgmask)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#cap.release()
#cv2.destroyAllWindows()

import numpy as np
import cv2


def img_processing(img):
    #t2 = time.time()
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    str_element_size=(16,4)
    kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    enhanced_gray=cv2.filter2D(imgGrayscale, -1, kernel)
    imgBlurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)# blur

    equal_histogram = cv2.equalizeHist(imgBlurred)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    sobelx = cv2.Sobel(equal_histogram, -1, 1,0) #Sobel of the equalized hist grayscale image for enhanced edges
    #se = cv2.getStructuringElement(cv2.MORPH_RECT, str_element_size)
    #t8 = time.time()
    morph_image = cv2.morphologyEx(sobelx,cv2.MORPH_OPEN,kernel2,iterations=10)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    h,sobel = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #sobel= cv2.adaptiveThreshold(sub_morp_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55,5)
    #gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)
    
    return sobel, equal_histogram

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




#imgOriginal = cv2.imread("CV/still.jpg")  
#imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
#imgBlurred = cv2.GaussianBlur(imgGrayscale, (3, 3), 0)# blur
#equal_histogram = cv2.equalizeHist(imgBlurred)
cap =cv2.VideoCapture('Ingi.mov') #cv2.VideoCapture('CV/MVI_6708.MOV') #cv2.VideoCapture('Ingi.mov')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.createBackgroundSubtractorMOG2()
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#firstframe=None
w,h,x,y=0,0,0,0
while(1):
    ret, frame = cap.read()
    #imgGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #kernel2 = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    #enhanced_gray=cv2.filter2D(imgGrayscale, -1, kernel2)

    #blur=cv2.medianBlur(imgGrayscale,3)
    #imgBlurred = cv2.GaussianBlur(blur, (7, 7), 0)# blur
    #erode=cv2.erode(fgmask,None,iterations=3)     #erosion to erase unwanted small contours
    #moments=cv2.moments(erode,True) 
    #grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurframe = cv2.GaussianBlur(grayframe, (3, 3), 0)# blur
    #equal_his = cv2.equalizeHist(blurframe)
    #med_blur = cv2.medianBlur(grayframe,3)
    #blur = cv2.GaussianBlur(med_blur,(7,7),0)
    #cv2.imshow('blurred image',blur)
    #sobelx = cv2.Sobel(equal_his, -1, 0,1) #Sobel of the equalized hist grayscale image for enhanced edges
    #se = cv2.getStructuringElement(cv2.MORPH_RECT, str_element_size)
    
    
    #kernel3= cv2.getStructuringElement(cv2.MORPH_RECT,(55,35))
    #tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel3)
    #_, bin_img = cv2.threshold(tophat,80,255,cv2.THRESH_BINARY)
    #kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #closed_img = cv2.morphologyEx(bin_img,cv2.MORPH_CLOSE,kernel4)
    
    
    #morph_image = cv2.morphologyEx(equal_his,cv2.MORPH_OPEN,kernel,iterations=10)
    #sub_morp_image = cv2.subtract(equal_his,morph_image)
    #sobel= cv2.adaptiveThreshold(sub_morp_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 43,2)
    #h,sobel = cv2.threshold(sub_morp_image,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('thre',closed_img)
    #fgmask=closed_img
    #fgmask,blurry=img_processing(frame)
    #can = cv2.Canny(equal_his,100,150)
    #cv2.imshow('can',can)
    #fgmask = fgbg.apply(equal_his,equal_histogram)
    fgmask = fgbg.apply(frame)
    
    #cv2.subtract(fgmask,)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    #if firstframe is None:
	#	firstframe = imgBlurred
	#	continue
    #frameDelta = cv2.absdiff(firstframe, fgmask)
#    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
   # processed=img_processing(frame,fgmask)
    h,fgmask = cv2.threshold(fgmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #fgmask= cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55,5)
    fg = cv2.bitwise_and(frame, frame, mask=fgmask)
    fg2,equalized=img_processing(fg)
    ## get second masked value (background) mask must be inverted
    #mask = cv2.bitwise_not(fgmask)
    #background = np.full(frame.shape, 255, dtype=np.uint8)
    #bk = cv2.bitwise_or(background, background, mask=mask)
    recx=500
    recy=600
    recw=800
    rech=700
    #recx=500
    #recy=425
    #recw=800
    #rech=700
    #framesearch=fgmask[recy:recy+rech,recx:recx+recw]
    framesearch=frame[150:1000,200:1200 ]
    cv2.imshow('fsaas',fg2)
    _,contours,_=cv2.findContours(fg2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.rectangle(frame, (recx,recy), (recx+1000, recy+850), (0,0,255), 6)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500 and area < 2000:
            [x,y,w,h] = cv2.boundingRect(contour)
            if w>4*h and w<5*h and y>150 and y<1000 and x>200 and x<1200: #w > 40 and w < 900 and h > 50 and h < 600:   ##Aspect ratio of the licence plates
                #ar=w/h
                crop_img = equalized[y:y+h,x:x+w]
                edgedensity=density(crop_img,w,h,x,y)
                if edgedensity>0.5:
                    ar=w/h
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 6)
                    cv2.putText(frame,'AR: %r' %ar, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)
    cv2.imshow('ssss',frame)
#    out.write(fgmask)
    #cv2.imshow('frame',fgmask)
    #cv2.imshow('bl',blurry)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()