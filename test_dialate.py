import cv2
import numpy as np


threshold_value = 150

_, multi_channel_data = cv2.imreadmulti('img_crop.tif', [], cv2.IMREAD_ANYCOLOR)
# print(ret.shape)
# print(len(multi_channel_data))



n_channels = 4
n_layers = len(multi_channel_data) // n_channels
img_height, img_width, _ = multi_channel_data[0].shape
all_img = np.zeros((n_channels, n_layers, img_height, img_width), dtype=np.uint8)

for ii, this_data in enumerate(multi_channel_data):
    all_img[ii%n_channels, ii//n_channels] = this_data[:,:,2]


# print(np.max(all_img[0,0]))
# print(np.max(multi_channel_data[0][:,:,2]))

cv2.imshow("Image Grid", multi_channel_data[0][:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()


image1 = all_img[0,0]
image2 = all_img[1,0]
# Apply thresholding to create binary images
_, binary_image1 = cv2.threshold(image1, threshold_value, 255, cv2.THRESH_BINARY)
_, binary_image2 = cv2.threshold(image2, threshold_value, 255, cv2.THRESH_BINARY)



# Load an image (replace 'your_image.jpg' with the actual file path)
image = np.stack((image1, np.zeros(image1.shape, dtype=np.uint8), np.zeros(image1.shape, dtype=np.uint8)), axis=2, dtype=np.uint8)
# _,image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
kernel3 = np.ones((3, 3), np.uint8)  # You can adjust the kernel size
kernel_norm = np.array([[0,0,1,0,0],
                        [0,2,2,2,0],
                        [1,2,5,2,1],
                        [0,2,2,2,0],
                        [0,0,1,0,0],
                        ], dtype=np.float32)/25
img3 = cv2.dilate(image, kernel3, iterations=1)
# img_norm = cv2.dilate(image, kernel_norm, iterations=1)
blurred = cv2.medianBlur(image, 7)
blurred_gau = cv2.GaussianBlur(image, (11, 11), 0)
_,result = cv2.threshold(blurred_gau, 50, 255, cv2.THRESH_BINARY)



kernel = np.ones((5,5),np.float32)/25
img_norm = cv2.filter2D(image,-1,kernel_norm)

cv2.imshow('Original Image', image)
cv2.waitKey(0)
# cv2.imshow('Img 3', img3)
# cv2.waitKey(0)
# cv2.imshow('Img norm', img_norm)
# cv2.waitKey(0)
# cv2.imshow('Img blur', blurred)
# cv2.waitKey(0)
cv2.imshow('Img blur', blurred_gau)
cv2.waitKey(0)
cv2.imshow('result', result)
cv2.waitKey(0)
