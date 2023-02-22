import cv2
import numpy as np

class PerspectiveTransformation:
    """ Đây là một lớp để chuyển đổi hình ảnh giữa góc nhìn thẳng và góc nhìn từ trên xuống

    Thuộc tính:
        src (np.array): Tọa độ của 4 điểm nguồn
        dst (np.array): Tọa độ của 4 điểm đến
        M (np.array): Ma trận để chuyển đổi hình ảnh từ chế độ xem trước sang chế độ xem từ trên xuống
        M_inv (np.array): Ma trận để chuyển đổi hình ảnh từ góc nhìn trên xuống góc nhìn thẳng
    """
    def __init__(self):
        """Khởi chạy PerspectiveTransformation."""
        self.src = np.float32([(550, 460),     # top-left
                               (150, 720),     # bottom-left
                               (1200, 720),    # bottom-right
                               (770, 460)])    # top-right
        self.dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """ Chụp ảnh ở góc nhìn thẳng và chuyển sang góc nhìn từ trên xuống

        Thông số:
            img (np.array): Một hình ảnh ở góc nhìn thẳng
            img_size (tuple): Kích thước của hình ảnh (chiều rộng, chiều cao)
            flags : cờ để sử dụng trong cv2.warpPerspective()

        Trả về:
            Image (np.array): Hình ảnh góc nhìn từ trên xuống
        """
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """ Chụp ảnh ở góc nhìn trên xuống và biến nó thành góc nhìn thẳng

        Thông số:
            img (np.array): Ảnh góc nhìn từ trên
            img_size (tuple): Kích thước của hình ảnh (chiều rộng, chiều cao)
            flags (int): cờ để sử dụng trong cv2.warpPerspective()

        Trả về:
            Image (np.array): Ảnh góc nhìn thẳng
        """
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)
