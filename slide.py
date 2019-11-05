from functools import partial
import os
import glob

import openslide as osli
import cv2 as cv
import numpy as np

from imageprocess import ImageProcess
from key_defs import *

class Slide:
    def __init__(self, conf):
        self.conf = conf
        self.detector = ImageProcess(self.conf)
        self.slide = self.load_image()
        self.initialize_windows()
        self.overview_factor, self.original_overview = self.generate_overview()
        self.overview = self.original_overview.copy()
        self.create_mask_window()

        self.pixels_per_square_mm = ((float(self.slide.properties[osli.PROPERTY_NAME_MPP_X]) * 1000) ** 2)

        self.current_iter = 0
        self.current_printed = False

        self.drawing = False
        self.draw_points = []
        self.overview_draw_points = []
        self.update_overview = True
        self.current_immune_cells = 0

        self.draw_counter = 0


    def initialize_windows(self):
        self.overview_window = "Overview"
        self.mask_window = "Mask"
        self.original_window = "Original"
        self.immunocells_window = "Detected immunocells"
        self.hed_window = "Hematoxylin-Eosin-DAB"
        self.conf.hed_window = "Hematoxylin-Eosin-DAB" 

        window_size = self.conf.options.window_size

        cv.namedWindow(self.overview_window, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.overview_window, window_size, window_size)
        cv.namedWindow(self.immunocells_window, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.immunocells_window, window_size, window_size)
        cv.setMouseCallback(self.overview_window, self.mouse_draw_overview)

        if self.conf.options.advanced:
            cv.namedWindow(self.mask_window, cv.WINDOW_NORMAL)
            cv.namedWindow(self.original_window, cv.WINDOW_NORMAL)
            cv.namedWindow(self.hed_window, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.hed_window, window_size, window_size)
            cv.resizeWindow(self.mask_window, window_size, window_size)
            cv.resizeWindow(self.original_window, window_size, window_size)


    def mouse_draw_overview(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.draw_counter += 1
            if self.draw_counter > 5:
                self.update_overview = True
                self.draw_counter = 0
            real_x = int((x * self.overview_factor) + int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_X]))
            real_y = int((y * self.overview_factor) + int(self.slide.properties[osli.PROPERTY_NAME_BOUNDS_Y]))
            self.draw_points.append([real_x, real_y])
            self.overview_draw_points.append([x,y])


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


    def get_region_selection(self, selection, level = 0):
        x,y,w,h = cv.boundingRect(selection)
        return self.slide.read_region((x,y), level, (w, h)).convert('RGB')


    def get_sections_selection(self, selection, level = 0):
        x,y,w,h = cv.boundingRect(selection)
        real_x = x - (x % 1000)
        real_y = y - (y % 1000)
        real_width = x + w + 1000 - ((x + w) % 1000)
        real_height = y + h + 1000 - ((y + h) % 1000)
        current_x = real_x
        current_y = real_y
        images = {}
        while current_y < real_height:
            while current_x < real_width:
                if current_x not in images:
                    images[current_x] = {}
                images[current_x][current_y] = self.slide.read_region((current_x, current_y), level, (1000,1000)).convert('RGB')
                current_x += 1000
            current_y += 1000
        return images


    def draw_overview_overlay(self):
        if not self.update_overview:
            return self.overview
        self.update_overview = False

        if len(self.overview_draw_points) > 4:
            contours = np.array(self.overview_draw_points).reshape((-1,1,2)).astype(np.int32)
            if self.drawing:
                cv.polylines(self.overview, [contours], False, (255,255,0),2)
            else:
                cv.drawContours(self.overview, [contours],0,(255,255,255),2)
        return self.overview


    def export_images(self, contour):
        if self.conf.options.create_segments:
            if not os.path.exists('selections'):
                os.mkdir('selections')
            dir_list = glob.glob('selections/*')
            target_dir = 100
            for i in range(1,100):
                if "selections/{}".format(str(i)) not in dir_list and target_dir == 100:
                    target_dir = str(i)
            os.mkdir("selections/{}".format(target_dir))
            images = self.get_sections_selection(contour)
            for x_value in images:
                for y_value in images[x_value]:
                    # save image as 'selections/<selection_num>/<x>_<y>.png'
                    images[x_value][y_value].save("selections/{}/{}_{}.png".format(target_dir, x_value, y_value))


    def __call__(self):
        while True:
            contour = np.array(self.draw_points).reshape((-1,1,2)).astype(np.int32)
            x,y,w,h = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if not self.current_printed and len(self.draw_points) > 4 and not self.drawing:
                self.export_images(contour)
                img = self.get_region_selection(contour)
                img = np.array(img)
                immune_cells, mask = self.detector.get_immune_cells(img)
                inside_cells = []
                for cell in immune_cells:
                    moment = cv.moments(cell)
                    center_x = int(moment['m10']/moment['m00']) + x
                    center_y = int(moment['m01']/moment['m00']) + y
                    if cv.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
                        inside_cells.append(cell)
                    immune_cells = inside_cells

                if not self.current_printed:
                    self.current_immune_cells = len(immune_cells)
                    print("Immune cells in selection: {}".format(len(immune_cells)))
                    self.current_printed = True
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                cvimg2 = img.copy()
                cvimg2 = cv.drawContours(cvimg2, immune_cells, -1, (0,255,0))
                updated_list = [[p[0] - x, p[1] - y] for p in self.draw_points]
                updated_contour = np.array(updated_list).reshape((-1,1,2)).astype(np.int32)
                cv.drawContours(cvimg2, [updated_contour], 0, (255,255,255), 4)

                if self.conf.options.advanced:
                    cv.imshow(self.mask_window, mask)
                    cv.imshow(self.original_window, img)
                cv.imshow(self.immunocells_window, cvimg2)

            over = self.draw_overview_overlay()
            cv.imshow(self.overview_window, over)

            if area > 0:
                cv.displayStatusBar(self.overview_window, "Immune cells in selection: {}, immune cells per square millimeter: {}".format(self.current_immune_cells, self.current_immune_cells / (area / self.pixels_per_square_mm)))

            key = cv.waitKeyEx(100)
            if key == KEY_ESC:
                return
            elif key == KEY_SPACE:
                if self.drawing:
                    self.current_printed = False
                    self.update_overview = True
                    print("Stopped drawing")
                    self.drawing = False
                    self.overview = self.original_overview.copy()
                else:
                    print("Started drawing")
                    self.drawing = True
                    self.draw_points = []
                    self.overview_draw_points = []
                    self.selected_regions = []
            elif key != -1:
                print("Button pressed {}".format(key))

    def load_image(self):
        return osli.OpenSlide(self.conf.options.input)

    def create_mask_window(self):
        update = partial(self.conf.update_configuration, self)
        cv.createTrackbar('Min hematoxylin',  'Mask', self.conf.options.hue_min, 255, partial(update, "hue_min"))
        cv.createTrackbar('Max hematoxylin',  'Mask', self.conf.options.hue_max, 255, partial(update, "hue_max"))
        cv.createTrackbar('Min eosin',  'Mask', self.conf.options.sat_min, 255, partial(update, "sat_min"))
        cv.createTrackbar('Max eosin',  'Mask', self.conf.options.sat_max, 255, partial(update, "sat_max"))
        cv.createTrackbar('Min DAB',  'Mask', self.conf.options.val_min, 255, partial(update, "val_min"))
        cv.createTrackbar('Max DAB',  'Mask', self.conf.options.val_max, 255, partial(update, "val_max"))
        cv.createTrackbar('Min area', 'Mask', self.conf.options.area_min, 5000, partial(update, "area_min"))
        cv.createTrackbar('Max area', 'Mask', self.conf.options.area_max, 5000, partial(update, "area_max"))
        cv.createTrackbar('Min circularity', 'Mask', self.conf.options.circularity, 100, partial(update, "circularity"))
