from math import pi
from functools import partial
import argparse

import openslide as osli
import cv2 as cv
import numpy as np

# Open file with openslide

# Configuration - from slide

class Configuration:
    def __init__(self):
        self.options = self._cmd_line_args()
    
    def _cmd_line_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hue_min', type=int, default=0)
        parser.add_argument('--sat_min', type=int, default=0)
        parser.add_argument('--val_min', type=int, default=10)
        parser.add_argument('--hue_max', type=int, default=140)
        parser.add_argument('--sat_max', type=int, default=255)
        parser.add_argument('--val_max', type=int, default=70)
        parser.add_argument('--area_min', type=int, default=200)
        parser.add_argument('--area_max', type=int, default=1500)
        parser.add_argument('--circularity', type=float, default=0.5)
        parser.add_argument('--input', required=True)
        return parser.parse_args()

    def update_configuration(self, name, value):
        options = vars(self.options)
        options[name] = value

class Main:
    def __init__(self):
        self.conf = Configuration()
        self.slide = self.load_image()
        self.iterations, self.it_colors = self.make_iterations()
        self.overview_factor, self.overview = self.generate_overview()

        self.auto_forward = True
        self.move_steps = 1

        self.current_iter = 0

    def make_iterations(self):
        iterations = []
        it_colors = []

        minx = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])
        miny = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])
        sizey = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_HEIGHT])
        sizex = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_WIDTH])
        for y in range(miny, miny+sizey, 1024):
            for x in range(minx, minx+sizex, 1024):
                iterations.append((x,y))
                it_colors.append((0,0,0))
        return iterations, it_colors

    def move(self):
        move_to = self.current_iter + self.move_steps
        if move_to < 0 or move_to > len(self.iterations):
            print("Tried to move to {}".format(move_to))
            raise IndexError("Reached end of slide set")
        return move_to

    def auto_move(self):
        if self.auto_forward:
            try:
                self.current_iter = self.move()
            except Exception as e:
                print("Out of bounds at iteration {}".format(self.current_iter))
                print(e)
                exit()

    def generate_overview(self):
        minx = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])
        miny = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])
        region = self.slide.read_region((minx, miny), 7, (1024,1024)).convert('RGB')
        overview = cv.cvtColor(np.array(region), cv.COLOR_RGB2BGR)
        overview_factor = self.slide.level_downsamples[7]
        return overview_factor, overview
        

    def get_region(self, iteration, size):
        coords = self.iterations[iteration]
        return self.slide.read_region(coords,0,(size,size)).convert('RGB')

    def get_immune_cells(self, segment):
        hsvimg = cv.cvtColor(np.array(segment), cv.COLOR_RGB2HSV)
        # Blur the image
        hsvimg = cv.bilateralFilter(hsvimg,5,75,75)
        # Generate mask based on HSV values
        hueLow = (self.conf.options.hue_min, self.conf.options.sat_min, self.conf.options.val_min)
        hueHigh = (self.conf.options.hue_max, self.conf.options.sat_max, self.conf.options.val_max)
        mask = cv.inRange(hsvimg, hueLow, hueHigh)
        # Contours based on mask
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # List of immune cells
        immune_cells = []
        # Generate immune cell candidates based on circularity and area
        for con in contours:
            area = cv.contourArea(con)
            perimiter = cv.arcLength(con, True)
            if perimiter == 0:
                continue
            circularity = 4*pi*(area/(perimiter**2))
            if self.conf.options.area_min < area < self.conf.options.area_max and circularity > self.conf.options.circularity:
                immune_cells.append(con)
        return immune_cells, mask
        
    def draw_overview_overlay(self):
        # Copy overview
        over = self.overview.copy()

        def min_overview(x, y):
            ret_x = int((x - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])) / self.overview_factor)
            ret_y = int((y - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])) / self.overview_factor)
            return (ret_x, ret_y)
        
        def max_overview(x, y):
            ret_x = int((x - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])) + 1024 / self.overview_factor) - 1
            ret_y = int((y - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])) + 1024 / self.overview_factor) - 1
            return (ret_x, ret_y)

        # Draw rectangles on overview based on how many immune cells are present
        for iteration, color in zip(self.iterations, self.it_colors):
            rec_x,rec_y = iteration
            cv.rectangle(over, min_overview(rec_x, rec_y), max_overview(rec_x, rec_y), color)
        # Draw rectangle on overview based on current segment
        current_x, current_y = self.iterations[self.current_iter]
        cv.rectangle(over, min_overview(current_x, current_y), max_overview(current_x, current_y), (0,255,0))
        return over
        

    def mainloop(self):
        while True:
            img = self.get_region(self.current_iter, 1024)
            # Optimization to discard blank images
            if(img.getbbox() == None):
                self.auto_move()

            # Convert from BGR to HSV
            immune_cells, mask = self.get_immune_cells(img)

            # Skip forward to next image if no immune cells were found
            if len(immune_cells) == 0:
                self.auto_move()
                continue
            else:
                # Print the number of immune cells
                print("Immune cells in image: {}".format(len(immune_cells)))
                # Set color on overview
                self.it_colors[self.current_iter] = (0,len(immune_cells)*2,0)
            # Show mask
            cv.imshow("Mask", mask)
            # Original image with immune cells outlined
            cvimg2 = np.array(img).copy()
            cvimg2 = cv.drawContours(cvimg2, immune_cells, -1, (0,255,0))
            over = self.draw_overview_overlay()

            # Show original image
            cv.imshow("Original", np.array(img))
            # Show immune cells on original image
            cv.imshow("Detected immunocells", cvimg2)
            # Show overview
            cv.imshow("Overview", over)

            # Event loop
            key = cv.waitKey(100)
            # Check for key presses
            if key == 27:
                return
            elif key == 49:
                self.going_forward = False
                self.auto_forward = True
                try:
                    self.current_iter = self.move()
                except:
                    return
            elif key == 50:
                self.going_forward = True
                self.auto_forward = True
                try:
                    self.current_iter = self.move()
                except:
                    return
            elif key >= 0:
                print("Button pressed {}".format(key))

    def load_image(self):
        return osli.OpenSlide(self.conf.options.input)

    def create_mask_window(self):
        cv.namedWindow('Mask')
        cv.createTrackbar('Min hue',  'Mask', self.conf.options.hue_min, 190, partial(self.conf.update_configuration, "hue_min"))
        cv.createTrackbar('Min sat',  'Mask', self.conf.options.sat_min, 255, partial(self.conf.update_configuration, "sat_min"))
        cv.createTrackbar('Min val',  'Mask', self.conf.options.val_min, 255, partial(self.conf.update_configuration, "val_min"))
        cv.createTrackbar('Max hue',  'Mask', self.conf.options.hue_max, 190, partial(self.conf.update_configuration, "hue_max"))
        cv.createTrackbar('Max sat',  'Mask', self.conf.options.sat_max, 255, partial(self.conf.update_configuration, "sat_max"))
        cv.createTrackbar('Max val',  'Mask', self.conf.options.val_max, 255, partial(self.conf.update_configuration, "val_max"))
        cv.createTrackbar('Min area', 'Mask', self.conf.options.area_min, 1000, partial(self.conf.update_configuration, "area_min"))
        cv.createTrackbar('Max area', 'Mask', self.conf.options.area_max, 1000, partial(self.conf.update_configuration, "area_max"))
        cv.createTrackbar('Min circularity', 'Mask', self.conf.options.circularity, 100, partial(self.conf.update_configuration, "circularity"))
        

m = Main()
m.mainloop()
        
# Close image file
m.slide.close()

# Close all windows
cv.destroyAllWindows()
