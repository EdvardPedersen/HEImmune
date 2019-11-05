import cv2 as cv

from configuration import Configuration
from slide import Slide
from pngslide import PNGSlide

if __name__ == "__main__":
    conf = Configuration()
    if conf.options.input.endswith(".mrxs"):
        m = Slide(conf)
    elif conf.options.input.endswith(".png"):
        m = PNGSlide(conf)
    else:
        raise RuntimeError("Wrong file specified")
    m()
