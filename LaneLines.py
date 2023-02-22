import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Lớp chứa thông tin về các làn đường được phát hiện.

    Thuộc tính:
        left_fit (np.array): Các hệ số của một đa thức phù hợp với làn đường bên trái
        right_fit (np.array): Các hệ số của đa thức phù hợp với đường làn bên phải
        parameters (dict): Từ điển chứa tất cả các tham số cần thiết cho pipeline
        debug (boolean): Gắn cờ cho chế độ gỡ lỗi/bình thường
    """
    def __init__(self):
        """Bắt đầu Lanelines.

        Thông số:
            left_fit (np.array): Các hệ số của đa thức phù hợp với làn đường bên trái
            right_fit (np.array): Các hệ số của đa thức phù hợp với làn bên phải
            binary (np.array): hình ảnh nhị phân
        """
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        # Số cửa sổ trượt
        self.nwindows = 9
        # Chiều rộng của cửa sổ +/- lề
        self.margin = 100
        # Số pixel tối thiểu được tìm thấy cho cửa sổ gần đây hơn
        self.minpix = 50

    def forward(self, img):
        """Chụp ảnh và phát hiện vạch làn đường.

        Thông số:
            img (np.array): Một hình ảnh nhị phân chứa các pixel có liên quan

        Trả về:
            Image (np.array): Một hình ảnh RGB chứa các điểm ảnh của làn đường và các chi tiết khác
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Trả lại tất cả pixel trong một cửa sổ cụ thể

        Thông số:
            center (tuple): tọa độ tâm cửa sổ
            margin (int): một nửa chiều rộng của cửa sổ
            height (int): chiều cao của cửa sổ

        Trả về:
            pixelx (np.array): tọa độ x của các pixel nằm bên trong cửa sổ
            pixely (np.array): tọa độ y của pixel nằm bên trong cửa sổ
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """ Trích xuất các tính năng từ một hình ảnh nhị phân

        Thông số:
            img (np.array): Một hình ảnh nhị phân
        """
        self.img = img
        # Chiều cao của cửa sổ - dựa trên n cửa sổ và hình dạng hình ảnh
        self.window_height = np.int(img.shape[0]//self.nwindows)

        # Xác định vị trí x và y của tất cả các pixel khác không trong ảnh
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Tìm pixel làn đường từ hình ảnh bị biến dạng nhị phân.

        Thông số:
            img (np.array): Một hình ảnh nhị phân bị cong 

        Trả về:
            leftx (np.array): tọa độ x của pixel làn đường bên trái
            lefty (np.array): tọa độ y của pixel làn bên trái
            rightx (np.array): tọa độ x của pixel làn bên phải
            righty (np.array): tọa độ y của pixel làn bên phải
            out_img (np.array): Một hình ảnh RGB sử dụng để hiển thị kết quả sau này.
        """
        assert(len(img.shape) == 2)

        # Tạo một hình ảnh đầu ra để vẽ và trực quan hóa kết quả
        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Vị trí hiện tại sẽ được cập nhật sau cho mỗi cửa sổ trong n cửa sổ 
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Tạo danh sách trống để nhận pixel làn trái và phải
        leftx, lefty, rightx, righty = [], [], [], []

        # Bước qua từng cửa sổ
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Nối các chỉ số này vào danh sách
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Tìm đường làn đường từ một hình ảnh và vẽ nó.

        Thông số:
            img (np.array): một hình ảnh nhị phân bị cong vênh 

        Trả về:
            out_img (np.array): một hình ảnh RGB có vạch làn đường được vẽ trên đó.
        """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Tạo các giá trị x và y để vẽ đồ thị
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Trực quan hóa
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        lR, rR, pos = self.measure_curvature()

        return out_img

    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 400
        H = 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key = self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, 400),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 255, 0),
            thickness=2)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 450),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.66,
            color=(255, 255, 255),
            thickness=2)

        return out_img

    def measure_curvature(self):
        ym = 30/720
        xm = 3.7/700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (bán kính cong)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos 
