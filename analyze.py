from math import pi, sqrt
from functools import partial, lru_cache
import argparse
import time

import openslide as osli
import cv2 as cv
import numpy as np
from skimage.color import rgb2hed
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity

KEY_RIGHT = 120
KEY_LEFT = 122
KEY_SPACE = 32
KEY_ESC = 27

class Configuration:
    def __init__(self):
        self.options = self._cmd_line_args()

    def _cmd_line_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hue_min', type=int, default=80)
        parser.add_argument('--hue_max', type=int, default=255)
        parser.add_argument('--sat_min', type=int, default=0)
        parser.add_argument('--sat_max', type=int, default=120)
        parser.add_argument('--val_min', type=int, default=0)
        parser.add_argument('--val_max', type=int, default=255)
        parser.add_argument('--area_min', type=int, default=190)
        parser.add_argument('--area_max', type=int, default=600)
        parser.add_argument('--circularity', type=int, default=65)
        parser.add_argument('--input', required=True)
        parser.add_argument('--size', type=int, default=2048)
        parser.add_argument('--window_size', type=int, default=1024)
        parser.add_argument('--overview_downsample', type=int, default=4)
        parser.add_argument('--advanced', action="store_true")
        parser.add_argument('--cuda', action="store_true")
        parser.add_argument('--slow', action="store_true")
        return parser.parse_args()

    def update_configuration(self, printer, name, value):
        printer.auto_forward = False
        printer.current_printed = False
        options = vars(self.options)
        options[name] = value

class Main:
    def __init__(self):
        self.conf = Configuration()
        if self.conf.options.cuda:
            print("enabling CUDA")
            self.km = __import__("libKMCUDA")
        self.slide = self.load_image()
        self.initialize_windows()
        self.overview_factor, self.original_overview = self.generate_overview()
        self.overview = self.original_overview.copy()
        self.create_mask_window()

        self.pixels_per_square_mm = ((float(self.slide.properties[osli.PROPERTY_NAME_MPP_X]) * 1000) ** 2)
        print(self.pixels_per_square_mm)

        self.current_iter = 0
        self.current_printed = False

        self.drawing = False
        self.draw_points = []
        self.overview_draw_points = []
        self.update_overview = True
        self.selection_start = False
        self.total_selection = 0
        self.current_immune_cells = 0
        self.output_selection = 0

        self.draw_counter = 0


    def initialize_windows(self):
        self.overview_window = "Overview"
        self.mask_window = "Mask"
        self.original_window = "Original"
        self.immunocells_window = "Detected immunocells"
        self.hed_window = "Hematoxylin-Eosin-DAB"

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

    @lru_cache(50)
    def get_region(self, iteration, size, level = 0):
        coords = self.iterations[iteration]
        real_size = int(size / self.slide.level_downsamples[level])
        return self.slide.read_region(coords,level,(real_size,real_size)).convert('RGB')

    def get_region_selection(self, selection, level = 0):
        x,y,w,h = cv.boundingRect(selection)
        return self.slide.read_region((x,y), level, (w, h)).convert('RGB')

    def get_immune_cells(self, image):
        # Color limits
        hueLow = (self.conf.options.hue_min, self.conf.options.sat_min, self.conf.options.val_min)
        hueHigh = (self.conf.options.hue_max, self.conf.options.sat_max, self.conf.options.val_max)

        # Full resolution follows
        hedimg = rgb2hed(image)
        hedimg[:,:,0] = rescale_intensity(hedimg[:,:,0])
        hedimg[:,:,1] = rescale_intensity(hedimg[:,:,1])
        hedimg[:,:,2] = rescale_intensity(hedimg[:,:,2])
        hsvimg = img_as_ubyte(hedimg)

        if self.conf.options.slow:
            # Constants for k-means
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            num_colors = 8

            # Initial attempt at finding labels
            cv.setRNGSeed(num_colors)
            resize_factor = 4

            # Setup full image
            slow_image = hsvimg.reshape((-1,3))
            slow_image = np.float32(slow_image)

            # Setup smaller image
            fast_image = cv.resize(hsvimg, None, fx = 1 / resize_factor, fy = 1 / resize_factor)
            fast_image_array = fast_image.reshape((-1,3))
            fast_image_array = np.float32(fast_image_array)

            # Generate labels on large image
            if self.conf.options.cuda:
                center, label = self.km.kmeans_cuda(slow_image, num_colors,tolerance=0.1, seed=4, device=0)
            else:
                # Generate labels on small image
                _, fast_label, fast_center = cv.kmeans(fast_image_array, num_colors, None, criteria, 5, cv.KMEANS_RANDOM_CENTERS)
                fast_label = cv.resize(fast_label, (1,fast_label.size * resize_factor * resize_factor), interpolation = cv.INTER_NEAREST)
                fast_center = np.multiply(fast_center,(resize_factor * resize_factor))
                _, label, center = cv.kmeans(slow_image, num_colors, fast_label, criteria, 1, cv.KMEANS_USE_INITIAL_LABELS + cv.KMEANS_PP_CENTERS, fast_center)

            # Update image with new color space
            center = np.uint8(center)
            res = center[label.flatten()]
            hsvimg = res.reshape((hsvimg.shape))

        # Filter on color space
        if self.conf.options.advanced:
            cv.imshow(self.hed_window, hsvimg)
            cv.waitKey(1)

        # Generate mask
        mask = cv.inRange(img_as_ubyte(hsvimg), hueLow, hueHigh)
        kernels = [np.ones((3,3), np.uint8),
                   cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))]

        cons = None
        for kernel in kernels:
            temp_mask = mask.copy()
            temp_mask = cv.morphologyEx(temp_mask, cv.MORPH_OPEN, kernel, iterations=5)
            contours, hierarchy = cv.findContours(temp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not cons:
                cons = contours
            else:
                cons.extend(contours)

        immune_cells = []
        for con in cons:
            area = cv.contourArea(con)
            perimiter = cv.arcLength(con, True)
            if perimiter == 0:
                continue
            circularity = 4*pi*(area/(perimiter**2))
            if self.conf.options.area_min < area < self.conf.options.area_max and circularity*100 > self.conf.options.circularity:
                immune_cells.append(con)

        for ic in immune_cells:
            for other_index, other_con in enumerate(immune_cells):
                if self.contour_overlap(ic, other_con):
                    immune_cells.pop(other_index)
        return immune_cells, mask


    def contour_overlap(self, con1, con2):
        if id(con1) == id(con2):
            return False
        bb1 = cv.boundingRect(con1)
        bb2 = cv.boundingRect(con2)
        dist_x = bb1[0] - bb2[0]
        dist_y = bb1[1] - bb2[1]

        if abs(dist_x) + abs(dist_y) > 10:
            return False
        return True


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

        if len(self.overview_draw_points) > 4:
            contours = np.array(self.overview_draw_points).reshape((-1,1,2)).astype(np.int32)
            if self.drawing:
                cv.polylines(self.overview, [contours], False, (255,255,0),2)
            else:
                cv.drawContours(self.overview, [contours],0,(255,255,255),2)
        return self.overview


    def mainloop(self):
        while True:
            contour = np.array(self.draw_points).reshape((-1,1,2)).astype(np.int32)
            x,y,w,h = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if not self.current_printed and len(self.draw_points) > 4 and not self.drawing:
                img = self.get_region_selection(contour)
                img = np.array(img)
                immune_cells, mask = self.get_immune_cells(img)
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
                    self.output_selection = 0
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


m = Main()
m.mainloop()

# Close image file
m.slide.close()

# Close all windows
cv.destroyAllWindows()
