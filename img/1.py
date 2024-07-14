import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage import color
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog

def chon():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (674, 450))  # Resize the image to a fixed size

        # Chuyển đổi sang dạng số thực
        img1 = img.astype(float)

        # Tạo ảnh bị nhiễu chuyển động cho từng kênh màu
        blurred_channels = []
        for channel in range(3):
            # Chuyển ảnh sang ảnh xám
            gray = color.rgb2gray(np.expand_dims(img1[:, :, channel], axis=2))
            gray = np.squeeze(gray)  # Remove the extra dimension

            # Tạo ảnh bị mờ do chuyển động
            LEN = 21
            THETA = 11
            PSF = np.zeros((LEN, LEN))
            PSF[int((LEN-1)/2), :] = np.ones(LEN)
            PSF = np.roll(PSF, -int((LEN-1)/2), axis=1)
            PSF = cv2.warpAffine(PSF, cv2.getRotationMatrix2D((LEN/2-0.5, LEN/2-0.5), THETA, 1.0), (LEN, LEN))
            PSF = PSF/np.sum(PSF)
            blurred = convolve2d(gray, PSF, mode='same', boundary='wrap')

            blurred_channels.append(blurred)

        # Tạo ảnh màu bị nhiễu chuyển động
        blurred_img = np.stack(blurred_channels, axis=2)

        # Hiển thị ảnh gốc
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title('Ảnh gốc')

        # Hiển thị ảnh bị mờ do chuyển động
        plt.figure()
        plt.imshow(blurred_img, cmap='gray')
        plt.axis('off')
        plt.title('Ảnh bị mờ do chuyển động')

        plt.show()

root = Tk()
root.title('Xử lý ảnh')
name = Label(root, font=('Arial', 14), text='Lọc ảnh mờ chuyển động')
name.pack()
button = Button(root, text="Chọn ảnh", command=chon)
button.pack()

root.mainloop()
