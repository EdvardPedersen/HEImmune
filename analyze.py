import openslide as osli
import cv2 as cv
import numpy as np
from math import pi
from functools import partial

slide = osli.OpenSlide("image/No_name_HE/1M01.mrxs")

minx = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_X])
miny = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])
sizey = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_HEIGHT])
sizex = int(slide.properties[osli.PROPERTY_NAME_BOUNDS_WIDTH])

hueLow = (0, 0, 10)
hueHigh = (140, 255, 70)
area_spec = (150, 250)
circularity_spec = 0.5

auto_forward = True


def performit(minx, miny, sizex, sizey, slide):
    global auto_forward
    iterations = []
    it_colors = []
    for y in range(miny, miny+sizey, 1024):
        for x in range(minx, minx+sizex, 1024):
            iterations.append((x,y))
            it_colors.append((0,0,0))
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

    def update_config(param, value):
        global hueLow, hueHigh, area_spec, circularity_spec, auto_forward
        auto_forward = False
        if param == "hue_min":
            hueLow = (value, hueLow[1], hueLow[2])
        elif param == "sat_min":
            hueLow = (hueLow[0], value, hueLow[2])
        elif param == "val_min":
            hueLow = (hueLow[0], hueLow[1], value)
        elif param == "hue_max":
            hueHigh = (value, hueHigh[1], hueHigh[2])
        elif param == "sat_max":
            hueHigh = (hueHigh[0], value, hueHigh[2])
        elif param == "val_max":
            hueHigh = (hueHigh[0], hueHigh[1], value)
        elif param == "area_min":
            area_spec = (value, area_spec[1])
        elif param == "area_max":
            area_spec = (area_spec[0], value)
        elif param == "circularity":
            circularity_spec = value/100

    hue_min = hueLow[0]
    sat_min = hueLow[1]
    val_min = hueLow[2]
    hue_max = hueHigh[0]
    sat_max = hueHigh[1]
    val_max = hueHigh[2]
    area_min = area_spec[0]
    area_max = area_spec[1]
    cir_min = int(circularity_spec*100)
    cv.namedWindow('Mask')
    cv.createTrackbar('Min hue',  'Mask', hue_min, 190, partial(update_config, "hue_min"))
    cv.createTrackbar('Min sat',  'Mask', sat_min, 255, partial(update_config, "sat_min"))
    cv.createTrackbar('Min val',  'Mask', val_min, 255, partial(update_config, "val_min"))
    cv.createTrackbar('Max hue',  'Mask', hue_max, 190, partial(update_config, "hue_max"))
    cv.createTrackbar('Max sat',  'Mask', sat_max, 255, partial(update_config, "sat_max"))
    cv.createTrackbar('Max val',  'Mask', val_max, 255, partial(update_config, "val_max"))
    cv.createTrackbar('Min area', 'Mask', area_min, 1000, partial(update_config, "area_min"))
    cv.createTrackbar('Max area', 'Mask', area_max, 1000, partial(update_config, "area_max"))
    cv.createTrackbar('Min circularity', 'Mask', cir_min, 100, partial(update_config, "circularity"))

    total_sum = 0
    while True:
        x,y = iterations[current_iter]
        img = slide.read_region((x,y),0,(1024, 1024)).convert('RGB')
        if(img.getbbox() == None):
            if auto_forward:
                try:
                    current_iter = move(current_iter, going_forward)
                except Exception as e:
                    return
                continue
        cvimg = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        hsvimg = cv.cvtColor(cvimg, cv.COLOR_BGR2HSV)
        hsvimg = cv.bilateralFilter(hsvimg,5,75,75)
        # blur for the mask?
        mask = cv.inRange(hsvimg, hueLow, hueHigh)
        mask_contours, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        immune_cells = []
        for con in contours:
            area = cv.contourArea(con)
            perimiter = cv.arcLength(con, True)
            if perimiter == 0:
                continue
            circularity = 4*pi*(area/(perimiter**2))
            if area_spec[0] < area < area_spec[1] and circularity > circularity_spec:
                immune_cells.append(con)
        if len(immune_cells) == 0:
            if auto_forward:
                try:
                    current_iter = move(current_iter, going_forward)
                except:
                    return
                if(auto_forward):
                    continue
        else:
            print("Immune cells in image: {}".format(len(immune_cells)))
            it_colors[current_iter] = (0,len(immune_cells)*2,0)
            total_sum += len(immune_cells)
            #print("Total {} in {} iterations".format(total_sum, current_iter))
        cv.imshow("Mask", mask_contours)
        cvimg2 = cvimg.copy()
        cvimg2 = cv.drawContours(cvimg2, immune_cells, -1, (0,255,0))
        over = overview.copy()
        for i in range(len(iterations)):
            rec_x,rec_y = iterations[i]
            color = it_colors[i]
            cv.rectangle(over, (int((rec_x-minx)/overview_factor),int((rec_y-miny)/overview_factor)), (int(((rec_x-minx)+1024)/overview_factor) - 1, int(((rec_y-miny)+1024)/overview_factor) - 1), color)
        cv.rectangle(over, (int((x-minx)/overview_factor),int((y-miny)/overview_factor)), (int(((x-minx)+1024)/overview_factor), int(((y-miny)+1024)/overview_factor)), (0,255,0))
        cv.imshow("Original", cvimg)
        cv.imshow("Detected immunocells", cvimg2)
        cv.imshow("Overview", over)
        key = cv.waitKey(100)
        if key == 27:
            return
        elif key == 49:
            going_forward = False
            auto_forward = True
            try:
                current_iter = move(current_iter, going_forward)
            except:
                return
        elif key == 50:
            going_forward = True
            auto_forward = True
            try:
                current_iter = move(current_iter, going_forward)
            except:
                return
        elif key >= 0:
            print("Button pressed {}".format(key))


performit(minx, miny, sizex, sizey, slide)

slide.close()

cv.destroyAllWindows()
