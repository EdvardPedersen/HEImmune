from math import pi

import cv2 as cv
import numpy as np
from skimage.color import rgb2hed
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity

class ImageProcess:
    def __init__(self, conf):
        self.conf = conf
        if self.conf.options.cuda:
            print("enabling CUDA")
            self.km = __import__("libKMCUDA")

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
            cv.imshow(self.conf.hed_window, hsvimg)
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
