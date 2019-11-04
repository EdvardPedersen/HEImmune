import argparse

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
        parser.add_argument('--create_segments', action="store_true")
        return parser.parse_args()

    def update_configuration(self, printer, name, value):
        printer.auto_forward = False
        printer.current_printed = False
        options = vars(self.options)
        options[name] = value
