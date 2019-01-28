from math import pi
from functools import partial, lru_cache
import argparse
from copy import deepcopy

import openslide as osli
import cv2 as cv
import numpy as np

class Configuration:
    def __init__(self):
        self.options = self._cmd_line_args()

    def _cmd_line_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hue_min', type=int, default=120)
        parser.add_argument('--sat_min', type=int, default=0)
        parser.add_argument('--val_min', type=int, default=60)
        parser.add_argument('--hue_max', type=int, default=140)
        parser.add_argument('--sat_max', type=int, default=255)
        parser.add_argument('--val_max', type=int, default=110)
        parser.add_argument('--area_min', type=int, default=200)
        parser.add_argument('--area_max', type=int, default=4000)
        parser.add_argument('--circularity', type=int, default=40)
        parser.add_argument('--input', required=True)
        parser.add_argument('--size', type=int, default=1024)
        parser.add_argument('--window_size', type=int, default=1024)
        parser.add_argument('--overview_downsample', type=int, default=5)
        parser.add_argument('--selection', type=bool, default=False)
        return parser.parse_args()

    def update_configuration(self, printer, name, value):
        printer.auto_forward = False
        printer.current_printed = False
        options = vars(self.options)
        options[name] = value

class Main:
    def __init__(self):
        self.conf = Configuration()
        self.slide = self.load_image()
        self.initialize_windows()
        self.iterations, self.it_colors = self.make_iterations()
        self.overview_factor, self.original_overview = self.generate_overview()
        self.overview = self.original_overview.copy()
        self.create_mask_window()

        self.auto_forward = True
        self.move_steps = 1

        self.current_iter = 0
        self.current_printed = False

        self.drawing = False
        self.draw_points = []
        self.update_overview = True


    def initialize_windows(self):
        self.overview_window = "Overview"
        self.mask_window = "Mask"
        self.original_window = "Original"
        self.immunocells_window = "Detected immunocells"

        window_size = self.conf.options.window_size

        cv.namedWindow(self.overview_window, cv.WINDOW_NORMAL)
        cv.namedWindow(self.mask_window, cv.WINDOW_NORMAL)
        cv.namedWindow(self.original_window, cv.WINDOW_NORMAL)
        cv.namedWindow(self.immunocells_window, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.overview_window, window_size, window_size)
        cv.resizeWindow(self.mask_window, window_size, window_size)
        cv.resizeWindow(self.original_window, window_size, window_size)
        cv.resizeWindow(self.immunocells_window, window_size, window_size)
        cv.setMouseCallback(self.overview_window, self.mouse_draw_overview)

    def make_iterations(self):
        iterations = []
        it_colors = []

        minx = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])
        miny = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])
        sizey = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_HEIGHT])
        sizex = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_WIDTH])
        for y in range(miny, miny+sizey, self.conf.options.size):
            for x in range(minx, minx+sizex, self.conf.options.size):
                iterations.append((x,y))
                it_colors.append((0,0,0))
        return iterations, it_colors

    def move(self):
        move_to = self.current_iter + self.move_steps
        if move_to < 0 or move_to > len(self.iterations):
            print("Tried to move to {}".format(move_to))
            raise IndexError("Reached end of slide set")
        self.current_printed = False
        self.update_overview = True
        return move_to

    def auto_move(self):
        if self.auto_forward:
            try:
                self.current_iter = self.move()
            except Exception as e:
                print("Out of bounds at iteration {}".format(self.current_iter))
                print(e)
                exit()

    def mouse_draw_overview(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.draw_points.append([x, y])
            print("Added point")

    def generate_overview(self):
        minx = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])
        miny = int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])
        level = self.conf.options.overview_downsample
        factor = self.slide.level_downsamples[level]
        width = int(float(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_WIDTH]) / factor)
        height = int(float(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_HEIGHT]) / factor)
        region = self.slide.read_region((minx, miny), level, (width,height)).convert('RGB')
        overview = cv.cvtColor(np.array(region), cv.COLOR_RGB2BGR)
        return factor, overview

    @lru_cache(50)
    def get_region(self, iteration, size):
        coords = self.iterations[iteration]
        return self.slide.read_region(coords,0,(size,size)).convert('RGB')

    def get_immune_cells(self, segment):
        hsvimg = cv.cvtColor(segment, cv.COLOR_RGB2HSV)
        hsvimg = cv.bilateralFilter(hsvimg,5,150,150)
        hueLow = (self.conf.options.hue_min, self.conf.options.sat_min, self.conf.options.val_min)
        hueHigh = (self.conf.options.hue_max, self.conf.options.sat_max, self.conf.options.val_max)
        mask = cv.inRange(hsvimg, hueLow, hueHigh)
        kernel = np.ones((3,3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        immune_cells = []
        for con in contours:
            area = cv.contourArea(con)
            perimiter = cv.arcLength(con, True)
            if perimiter == 0:
                continue
            circularity = 4*pi*(area/(perimiter**2))
            if self.conf.options.area_min < area < self.conf.options.area_max and circularity*100 > self.conf.options.circularity:
                immune_cells.append(con)
        return immune_cells, mask

    def draw_overview_overlay(self):
        if not self.update_overview:
            return self.overview
        self.update_overview = False
        def min_overview(x, y):
            ret_x = int((x - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X])) / self.overview_factor)
            ret_y = int((y - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y])) / self.overview_factor)
            return (ret_x, ret_y)

        def max_overview(x, y):
            ret_x = int((x - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X]) + self.conf.options.size) / self.overview_factor) - 1
            ret_y = int((y - int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y]) + self.conf.options.size) / self.overview_factor) - 1
            return (ret_x, ret_y)

        # Draw rectangles on overview based on how many immune cells are present
        for iteration, color in zip(self.iterations, self.it_colors):
            rec_x,rec_y = iteration
            cv.rectangle(self.overview, min_overview(rec_x, rec_y), max_overview(rec_x, rec_y), color)
        # Draw rectangle on overview based on current segment
        current_x, current_y = self.iterations[self.current_iter]
        cv.rectangle(self.overview, min_overview(current_x, current_y), max_overview(current_x, current_y), (0,255,0))
        if not self.drawing and len(self.draw_points) > 4:
            contours = np.array(self.draw_points).reshape((-1,1,2)).astype(np.int32)
            cv.drawContours(self.overview, [contours],0,(255,255,255),1)
        return self.overview


    def mainloop(self):
        while True:
            if not self.current_printed:
                img = self.get_region(self.current_iter, self.conf.options.size)
                if(img.getbbox() == None):
                    self.auto_move()
                    continue

                img = np.array(img)
                immune_cells, mask = self.get_immune_cells(img)

                if len(immune_cells) == 0 and self.auto_forward:
                    self.auto_move()
                    continue
                else:
                    if not self.current_printed:
                        print("Immune cells in image: {}".format(len(immune_cells)))
                        self.current_printed = True
                    self.it_colors[self.current_iter] = (0,len(immune_cells)*2,0)
                cvimg2 = img.copy()
                cvimg2 = cv.drawContours(cvimg2, immune_cells, -1, (0,255,0))

                cv.imshow(self.mask_window, mask)
                cv.imshow(self.original_window, img)
                cv.imshow(self.immunocells_window, cvimg2)

            over = self.draw_overview_overlay()
            cv.imshow(self.overview_window, over)

            key = cv.waitKey(100)
            if key == 27: # ESC
                return
            elif key == 49: # 1
                self.move_steps = -1
                self.auto_forward = True
                try:
                    self.current_iter = self.move()
                except:
                    return
            elif key == 50: # 2
                self.move_steps = 1
                self.auto_forward = True
                try:
                    self.current_iter = self.move()
                except:
                    return
            elif key == 46: # .
                self.current_printed = False
                if self.drawing:
                    self.update_overview = True
                    print("Stopped drawing")
                    self.drawing = False
                else:
                    self.update_overview = True
                    self.overview = self.original_overview.copy()
                    print("Started drawing")
                    self.drawing = True
                    self.draw_points = []
            elif key >= 0:
                print("Button pressed {}".format(key))

    def load_image(self):
        return osli.OpenSlide(self.conf.options.input)

    def create_mask_window(self):
        update = partial(self.conf.update_configuration, self)
        cv.createTrackbar('Min hue',  'Mask', self.conf.options.hue_min, 190, partial(update, "hue_min"))
        cv.createTrackbar('Min sat',  'Mask', self.conf.options.sat_min, 255, partial(update, "sat_min"))
        cv.createTrackbar('Min val',  'Mask', self.conf.options.val_min, 255, partial(update, "val_min"))
        cv.createTrackbar('Max hue',  'Mask', self.conf.options.hue_max, 190, partial(update, "hue_max"))
        cv.createTrackbar('Max sat',  'Mask', self.conf.options.sat_max, 255, partial(update, "sat_max"))
        cv.createTrackbar('Max val',  'Mask', self.conf.options.val_max, 255, partial(update, "val_max"))
        cv.createTrackbar('Min area', 'Mask', self.conf.options.area_min, 5000, partial(update, "area_min"))
        cv.createTrackbar('Max area', 'Mask', self.conf.options.area_max, 5000, partial(update, "area_max"))
        cv.createTrackbar('Min circularity', 'Mask', self.conf.options.circularity, 100, partial(update, "circularity"))


m = Main()
m.mainloop()

# Close image file
m.slide.close()

# Close all windows
cv.destroyAllWindows()
