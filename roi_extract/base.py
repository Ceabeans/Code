import itertools
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import os
import shutil

import cv2
from colorlog import ColoredFormatter
from matplotlib import pyplot as plt
import toml

current_dir_root = os.path.dirname(os.path.abspath(__file__))
test_dir_root_dir = os.path.join(current_dir_root, "test")
hand_key_points_dir = os.path.join(current_dir_root, "palm_roi_ext", "hand_key_points")
hand_segment_dir = os.path.join(current_dir_root, "palm_roi_ext", "hand_segment")
test_dir_root = os.path.join(current_dir_root, "test")
config_toml = toml.load(os.path.join(current_dir_root, "config.toml"))



class ShowImage(object):
    def show_image(self, name, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.title(name)
        plt.axis('off')
        plt.show()


def hex_to_rgb(hex_color):
    """
    Convert a hexadecimal color string to RGB format  
    :param hex_color: Hexadecimal color string, for example "#fdfbfb"  
    :return: (R, G, B) tuple
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


import logging
def setup_logger():
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

mylogger = setup_logger()

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
        else:
            print(f"File {file_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: unable to delete {file_path}.")
    except FileNotFoundError:
        print(f"File not found: {file_path}.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")

