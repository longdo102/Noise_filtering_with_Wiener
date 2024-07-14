import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage import color, img_as_float
from skimage.restoration import wiener
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def chon():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Đọc ảnh từ tệp được chọn
        img = cv2.imread(file_path)
        img = img_as_float(img)
        
        # Chuyển ảnh sang ảnh xám
        gray = color.rgb2gray(img)
        
        # Hiển thị ảnh gốc
        plt.figure()
        plt.imshow(gray, cmap='gray')
        plt.title('Ảnh xám')
        
        # Tạo ảnh bị mờ do chuyển động
        LEN = 21
        THETA = 11
        PSF = np.zeros((LEN, LEN))
        PSF[int((LEN-1)/2), :] = np.ones(LEN)
        PSF = np.roll(PSF, -int((LEN-1)/2), axis=1)
        PSF = cv2.warpAffine(PSF, cv2.getRotationMatrix2D((LEN/2-0.5, LEN/2-0.5), THETA, 1.0), (LEN, LEN))
        PSF = PSF/np.sum(PSF)
        blurred = convolve2d(gray, PSF, mode='same', boundary='wrap')
        
        # Hiển thị ảnh bị mờ do chuyển động
        plt.figure()
        plt.imshow(blurred, cmap='gray')
        plt.title('Ảnh bị mờ do chuyển động')
        
        # Khôi phục ảnh bị mờ do chuyển động bằng phương pháp Wiener
        wnr1 = wiener(blurred, PSF, 0.01)

        # Chuyển đổi ảnh xám sang ảnh màu
        img_colorized = cv2.cvtColor(wnr1, cv2.COLOR_GRAY2BGR)
        # Lấy kích thước của ảnh màu
        height, width, _ = img.shape

        # Resize ảnh xám để khớp kích thước với ảnh màu
        img_gray_resized = cv2.resize(img_colorized, (width, height))

        # Kết hợp ảnh xám với ảnh màu
        img_output = cv2.addWeighted(img, 0.7, img_gray_resized, 0.3, 0)

        # Hiển thị ảnh khôi phục
        cv2.imshow("Anh mau", img_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Hiển thị tất cả các ảnh
        plt.show()

root = Tk()
root.title('Xử lý ảnh')
name = Label(root, font=('Arial',14), text='Lọc ảnh mờ chuyển động')
name.pack()
button = Button(root, text="Chọn ảnh", command=chon)
button.pack()

root.mainloop()
