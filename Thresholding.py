import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    """ Lớp này dùng để trích xuất các pixel có liên quan trong một hình ảnh.
    """
    def __init__(self):
        """ Khởi chạy Thresholding."""
        pass

    def forward(self, img):
        """ Chụp ảnh và trích xuất tất cả các pixel có liên quan.

        Thông số:
            img (np.array): hình ảnh đầu vào

        Trả về:
            binary (np.array): Một hình ảnh nhị phân đại diện cho tất cả các vị trí của pixel liên quan.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]

        right_lane = threshold_rel(l_channel, 0.8, 1.0)
        right_lane[:,:750] = 0

        left_lane = threshold_abs(h_channel, 20, 30)
        left_lane &= threshold_rel(v_channel, 0.7, 1.0)
        left_lane[:,550:] = 0

        img2 = left_lane | right_lane

        return img2
