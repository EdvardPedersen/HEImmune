import cv2 as cv

from imageprocess import ImageProcess
from key_defs import *

class PNGSlide:
    def __init__(self, conf):
        image = cv.imread(conf.options.input)
        conf.hed_window = "HED"
        detector = ImageProcess(conf)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        immune_cells, _, gs_proc = detector.get_immune_cells(rgb_image)
        self.gs_proc = gs_proc
        self.processed = cv.drawContours(image, immune_cells, -1, (0,255,0))
        print("Immune cells in image: {}".format(len(immune_cells)))
        if conf.options.output_file:
            cv.imwrite(conf.options.output_file, self.processed)

    def __call__(self):
        while True:
            cv.imshow("processed", self.processed)
            cv.imshow("cont", self.gs_proc)
            key = cv.waitKeyEx(100)
            if key == KEY_ESC:
                return
            elif key != -1:
                print("Button pressed {}".format(key))
