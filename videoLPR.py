
import numpy as np
import cv2
import time
def img_processing(img):
    #t2 = time.time()
    imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    str_element_size=(16,4)
    kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    enhanced_gray=cv2.filter2D(imgGrayscale, -1, kernel)
    blur=cv2.medianBlur(imgGrayscale,5)
    imgBlurred = cv2.GaussianBlur(blur, (7, 7), 0)# blur

    equal_histogram = cv2.equalizeHist(imgBlurred)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    sobelx = cv2.Sobel(equal_histogram, -1, 1,0) #Sobel of the equalized hist grayscale image for enhanced edges

    morph_image = cv2.morphologyEx(sobelx,cv2.MORPH_OPEN,kernel2,iterations=10)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    h,sobel = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
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



cap= cv2.VideoCapture('Ingi.MOV')#cv2.VideoCapture('CV/MVI_6708.MOV') #cv2.VideoCapture('Ingi.mov')
#cap.set(cv2.CAP_PROP_FPS, 20)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,6))
fgbg = cv2.createBackgroundSubtractorMOG2()
cv2.namedWindow("license",cv2.WINDOW_NORMAL)
cv2.namedWindow("fg",cv2.WINDOW_NORMAL)
cv2.namedWindow("fg2",cv2.WINDOW_NORMAL)
cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')                         ###### Write to file a Xvid
#out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (1920,1080))   ###### Write the video to avi file  ##Use out parameter to save video

w,h,x,y=0,0,0,0
while(1):
    ret, frame = cap.read()
    start = time.time()
    #frame = cv2.resize(frame,None,fx=0.5, fy=0.5)
    #frame=frame[300:1080,300:1200]
  
    fgmask = fgbg.apply(frame)


    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)






    h,fgmask = cv2.threshold(fgmask,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #fgmask = cv2.dilate(fgmask, None, iterations=2)
    fg = cv2.bitwise_and(frame, frame, mask=fgmask)

    fg2,equalized=img_processing(fg)



    #framesearch=fgmask[recy:recy+rech,recx:recx+recw]
    ####### Ingi.MOV ->>>frame[200:1080,300:1100]
    ymin=300
    ymax=960
    xmin=350
    xmax=1700

    ####### MVI_6712.MOV
    #ymin=370/2
    #ymax=1080/2
    #xmin=700/2
    #xmax=1200/2



    #f_search=fg2[ymin:ymax,xmin:xmax]  #Window to search for LP
    cv2.imshow('fg',fg) #Foreground before proc

    cv2.imshow('fg2',fg2)
    _,contours,_=cv2.findContours(fg2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,0,255), 2)
    for contour in contours:
        area = cv2.contourArea(contour)

        #####
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
       
        #####
        if area > 1100 and area < 2500:#if area > 400 and area < 2200:
            [x,y,w,h] = cv2.boundingRect(contour)
            if w>3.7*h and w<5.3*h and y>ymin and y<ymax and x>xmin and x<xmax: ####and y>150 and y<1000 and x>200 and x<1200:   ##Aspect ratio of the licence plates
                crop_img = equalized[y:y+h,x:x+w]
                edgedensity=density(crop_img,w,h,x,y)
                if edgedensity>0.5:
                    plate = Rect(frame,rect,box)
                    ar=w/h
                    cropped=frame[y:y+h,x:x+w]
                    cv2.drawContours(frame,[box],0,(255,0,0),2)
                    #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                    cv2.putText(frame,'Area: %r' %area, (x+40,y+h+30), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
                    #cv2.putText(frame,'area: %r' %area, (20,30), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
                    cv2.imshow('license',plate)
                    #print rect[2]

                    #xypts.append([x,y])
                    #M = cv2.moments(fg2)
                    #print M
                    #cv2.circle(frame,(x,y),10,(0,255,0),-1)
                    #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    

            
    
    end = time.time()
    seconds = end - start
    FPS=int(1/seconds)
    #print "FPS : {0} ".format(float(1/seconds))
    #print seconds
    cv2.putText(frame,'FPS: %r' %FPS, (40,40), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 2)
    #fps  = num_frames / seconds;
    cv2.imshow('Original',frame)
    #print "Estimated frames per second : {0}".format(fps);
    #out.write(frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:

        break
out.release()
cap.release()
cv2.destroyAllWindows()