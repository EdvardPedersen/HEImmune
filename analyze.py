import openslide as osli
import cv2 as cv
import numpy as np
from math import pi

slide = osli.OpenSlide("CMU-1.mrxs")

minx = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_X])
miny = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])
sizey = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_HEIGHT])
sizex = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_WIDTH])

hueLow = (0, 0, 10)
hueHigh = (180, 255, 120)


def performit(minx, miny, sizex, sizey, hueLow, hueHigh, slide):
    iterations = []
    for y in range(miny, miny+sizey, 1024):
        for x in range(minx, minx+sizex, 1024):
            iterations.append((x,y))
    num = 0
    detector = cv.ORB.create()
    current_iter = 0
    going_forward = True
    def move(current_iter, going_forward):
        if going_forward:
            current_iter += 1
        else:
            current_iter -= 1
        if current_iter >= len(iterations):
            current_iter = len(iterations) - 1
            raise IndexError("Reached end")
        elif current_iter < 0:
            current_iter = 0
            raise IndexError("Reached start")
        return current_iter

    overview = cv.cvtColor(np.array(slide.read_region((minx, miny), 7, (1024,1024)).convert('RGB')), cv.COLOR_RGB2BGR)
    overview_factor = slide.level_downsamples[7]

    total_sum = 0
    while True:
        x,y = iterations[current_iter]
        num += 1
        if num % 100 == 0:
            print("Num: {} : {} {}".format(num, x, y))
        img = slide.read_region((x,y),0,(1024, 1024)).convert('RGB')
        if(img.getbbox() == None):
            try:
                current_iter = move(current_iter, going_forward)
            except:
                return
            continue
        cvimg = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        hsvimg = cv.cvtColor(cvimg, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsvimg, hueLow, hueHigh)
        mask_contours, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        immune_cells = []
        for con in contours:
            area = cv.contourArea(con)
            perimiter = cv.arcLength(con, True)
            if perimiter == 0:
                continue
            circularity = 4*pi*(area/(perimiter**2))
            if 50 < area < 500 and circularity > 0.8:
                immune_cells.append(con)
        if len(immune_cells) == 0:
            try:
                current_iter = move(current_iter, going_forward)
            except:
                return
            continue
        else:
            print("Immune cells in image: {}".format(len(immune_cells)))
            total_sum += len(immune_cells)
            print("Total {} in {} iterations".format(total_sum, current_iter))
        cv.imshow("Mask", mask_contours)
        cvimg2 = cvimg.copy()
        cvimg2 = cv.drawContours(cvimg2, immune_cells, -1, (0,255,0))
        over = overview.copy()
        cv.rectangle(over, (int((x-minx)/overview_factor),int((y-miny)/overview_factor)), (int(((x-minx)+1024)/overview_factor), int(((y-miny)+1024)/overview_factor)), (0,255,0))
        cv.imshow("Original", cvimg)
        cv.imshow("Detected immunocells", cvimg2)
        cv.imshow("Overview", over)
        key = cv.waitKey()
        if key == 27:
            return
        elif key == 81:
            going_forward = False
            try:
                current_iter = move(current_iter, going_forward)
            except:
                return
        elif key == 83:
            going_forward = True
            try:
                current_iter = move(current_iter, going_forward)
            except:
                return


performit(minx, miny, sizex, sizey, hueLow, hueHigh, slide)

slide.close()

cv.destroyAllWindows()
