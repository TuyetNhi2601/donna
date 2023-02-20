import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    """ Lớp hiệu chỉnh máy ảnh bằng hình ảnh bàn cờ.

    Thuộc tính:
        mtx (np.array): Ma trận máy ảnh 
        dist (np.array): Hệ số biến dạng
    """
    def __init__(self, image_dir, nx, ny, debug=False):
        """ Khởi động CameraCalibration.

        Thông số:
            image_dir (str): đường dẫn đến thư mục chứa ảnh bàn cờ
            nx (int): chiều rộng của bàn cờ (số ô vuông)
            ny (int): chiều cao của bàn cờ (số ô vuông)
        """
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []
        
        # Tọa độ các góc bàn cờ trong không gian 3D
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Đi qua tất cả các hình ảnh bàn cờ
        for f in fnames:
            img = mpimg.imread(f)

            # Chuyển đổi sang hình ảnh thang độ xám
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Tìm góc bàn cờ
            ret, corners = cv2.findChessboardCorners(img, (nx, ny))
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Không thể hiệu chỉnh máy ảnh")

    def undistort(self, img):
        """ Trả lại hình ảnh không bị biến dạng.

        Thông số:
            img (np.array): hình ảnh đầu vào

        Trả về:
            Image (np.array): Hình ảnh không bị biến dạng
        """
        # Chuyển đổi sang hình ảnh thang độ xám
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
