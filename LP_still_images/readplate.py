import cv2
from PIL import Image
from pytesseract import image_to_string

X,Y,W,H = 0,0,0,0
plate = []
temp = []
licenceplate = []
scaling = 290

def readplate(crop_img):
    licenceplate = []
    plate = []
    # Image scaling to have consistent text size
    r = float(scaling) / crop_img.shape[1]
    dim = (int(scaling), int(crop_img.shape[0] * r))
    crop_img = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
    
    # Converting image to grayscale, bluring and thersholding to find contours in image
    crop_grayscale = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray_crop = crop_grayscale
    gray_crop = cv2.medianBlur(gray_crop, 3)
    gray_crop = cv2.adaptiveThreshold(gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43,2)
    _,contourss,_ = cv2.findContours(gray_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Taking all countours that were found and looking for text
    for contour in contourss:
        smallarea = cv2.contourArea(contour)            # Calculating the area of each contour

        # If contour area is within set boundaries it cuts the countour out of the image 
        if smallarea > 200 and smallarea < 1500 :
            [X,Y,W,H] = cv2.boundingRect(contour)       # find x,y coordinates of the corners of the contour
            crop_smallimg = crop_img[Y:Y+H,X:X+W]       # crop the contour out of original image
            height,width,_ = crop_smallimg.shape        # find the height and width of the contour

            # If the countour height is whithin set boundaries the PYtesseract is used to find text
            if height>40 and height<50:
                img = cv2.cvtColor(crop_smallimg, cv2.COLOR_BGR2GRAY)                   # Convert croped image to grayscale
                _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)      # Threshold croped image
                crop_smallimg = cv2.GaussianBlur(img, (3,3), 0)                         # Bluring croped image using gaussian blur
                img_newsmall = Image.fromarray(crop_smallimg)                           # Create image from array
                cv2.rectangle(crop_img, (X,Y), (X+W, Y+H), (255,0,255), 2)              # Draw a reactangle on original image to show the contour
                temp = image_to_string(img_newsmall, lang='eng',config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 -psm 10")       
                # Read text from image using PYtessereact using "whitelist" function to skip all lowercase letters and other symbols
                # PYtesseract PSM modes to change what PYtesseract is looking for
                            # Page segmentation modes:
                            # 0    Orientation and script detection (OSD) only.
                            # 1    Automatic page segmentation with OSD.
                            # 2    Automatic page segmentation, but no OSD, or OCR.
                            # 3    Fully automatic page segmentation, but no OSD. (Default)
                            # 4    Assume a single column of text of variable sizes.
                            # 5    Assume a single uniform block of vertically aligned text.
                            # 6    Assume a single uniform block of text.
                            # 7    Treat the image as a single text line.
                            # 8    Treat the image as a single word.
                            # 9    Treat the image as a single word in a circle.
                            #10    Treat the image as a single character.
                            #11    Sparse text. Find as much text as possible in no particular order.
                            #12    Sparse text with OSD.
                            #13    Raw line. Treat the image as a single text line,

                if temp:        # If text is found it is put into a list
                    plate.append([X,temp])
    plate.sort()        # The list of text that has been found is sorted from left ro right
    if len(plate) == 7:         # If the list has 7 letters it is assumed that it has detected the iclandic flag on the number plate and it is removed
        remove = plate.pop(0)

    if len(plate) == 6:     # If the list has 6 letters it is assumed that it has detected the inspection sticker and it is removed
        remove = plate.pop(2)

    
    for i in range(len(plate)):             # Put all letters in a list to form the number on the numberplate
        licenceplate.append(plate[i][1])
    licenceplate = "".join(licenceplate)
    license = licenceplate
    del licenceplate
    del plate
    cv2.imshow('Licence plate contours',crop_img)       # Show original image with marked text contours
       
    return license


