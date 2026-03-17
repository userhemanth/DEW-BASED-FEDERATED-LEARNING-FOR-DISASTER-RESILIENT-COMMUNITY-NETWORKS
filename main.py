import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def test_environment():
    print("Python version:", torch.__version__)
    print("NumPy version:", np.__version__)
    print("OpenCV version:", cv2.__version__)
    print("Pillow version:", Image.__version__)
    print("Environment is ready!")

if __name__ == "__main__":
    test_environment()


from utils.helpers import greet
greet()
