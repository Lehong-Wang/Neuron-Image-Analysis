import cv2
import numpy as np
import matplotlib.pyplot as plt 

# img = cv2.imread("img.tif")
# plt.imshow(img[:,:,0], cmap="gray")
# plt.show()
# plt.imshow(img[:,:,1], cmap="gray")
# plt.show()
# plt.imshow(img[:,:,2], cmap="gray")
# plt.show()
# print(img.shape)

# img_array = np.array(img)
# print(img_array.shape)
# print(img_array)


threshold_value = 150

_, multi_channel_data = cv2.imreadmulti('img_crop.tif', [], cv2.IMREAD_ANYCOLOR)
# print(ret.shape)
# print(len(multi_channel_data))



# print(np.all(multi_channel_data[0][:,:,2] == 0))


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

# # img_3 = np.repeat(all_img[0,0][:, :, np.newaxis], 3, axis=2)


# cv2.imshow("Image Grid 2", all_img[0,0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.plot(all_img[0,0])
# plt.show()

# canvas_height = img_height * 4  # 4 rows
# canvas_width = img_width * 6   # 6 columns
# canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)


# canvas = np.reshape(all_img, ( n_layers, n_channels * img_height, img_width))
# canvas = np.reshape(canvas, (n_channels * img_height, n_layers * img_width))
# print(all_img.shape)
# print(canvas.shape)
# small = cv2.resize(canvas, (n_channels * img_height, n_layers * img_width))
# print(small.shape)

# cv2.imshow("All IMG", small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # exit()

# # Read the first image to determine its size
# first_image = all_img[0]
# # img_height, img_width, _ = first_image.shape
# img_height, img_width = (120,80)
# # Calculate the canvas size based on the size of the first image
# canvas_height = img_height * 4  # 4 rows
# canvas_width = img_width * 6   # 6 columns

# # Create a blank canvas for the grid
# canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# # Loop through the images and place them on the canvas
# for i, this_img in enumerate(img):
#     col, row = divmod(i, 4)
#     # img = cv2.imread(image_file)
#     resized_img = cv2.resize(this_img, (img_width, img_height))  # Resize the image to match the first image
#     canvas[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = resized_img

# # Display the grid
# cv2.imshow("Image Grid", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# image = all_img[1,1]

# # Set a threshold value (replace '128' with your desired threshold)
# threshold_value = 128

# # Apply thresholding
# _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# # Display the original and thresholded images
# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Thresholded Image', thresholded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



image1 = all_img[0,0]
image2 = all_img[1,0]
# Apply thresholding to create binary images
_, binary_image1 = cv2.threshold(image1, threshold_value, 255, cv2.THRESH_BINARY)
_, binary_image2 = cv2.threshold(image2, threshold_value, 255, cv2.THRESH_BINARY)

# Perform bitwise AND operation
result = cv2.bitwise_and(binary_image1, binary_image2)
result = np.repeat(result[:, :, np.newaxis], 3, axis=2)


# Display the original binary images and the result
cv2.imshow('Binary Image 1', binary_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Binary Image 2', binary_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Define a kernel for erosion and dilation
kernel5 = np.ones((5, 5), np.uint8)  # You can adjust the kernel size
kernel9 = np.ones((11, 11), np.uint8)  # You can adjust the kernel size
kernel1 = np.ones((11, 11), np.uint8)  # You can adjust the kernel size

# Erosion
dilated_image = cv2.dilate(result, kernel5, iterations=1)
eroded_image = cv2.erode(dilated_image, kernel9, iterations=1)

# Dilation
dilated_image = cv2.dilate(eroded_image, kernel1, iterations=1)



cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Dilated Image', dilated_image)
# print("Dilated Image shape:", dilated_image.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()







# # Set up the blob detector parameters
# params = cv2.SimpleBlobDetector_Params()

# # # Filter by Area.
# params.filterByArea = False
# # params.minArea = 10  # Adjust this value based on your specific case.
# params.filterByCircularity = False 

# # Create a blob detector with the specified parameters
# detector = cv2.SimpleBlobDetector_create(params)

# # Detect blobs
# keypoints = detector.detect(dilated_image)
# print(keypoints)

# # Draw the keypoints on the original image
# image_with_keypoints = cv2.drawKeypoints(dilated_image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Display the original image and the one with keypoints
# cv2.imshow('Original Image', dilated_image)
# cv2.imshow('Image with Keypoints', image_with_keypoints)

# # Extract and print the center positions of the blobs
# for i, kp in enumerate(keypoints):
#     print(f"Blob {i + 1}: Center Position ({kp.pt[0]}, {kp.pt[1]})")

# # Wait for a key event and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # Draw blobs on our image as red circles 
# blank = np.zeros((1, 1))  
# blobs = cv2.drawKeypoints(dilated_image, keypoints, blank, (0, 0, 255), 
#                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
  
# number_of_blobs = len(keypoints) 
# text = "Number of Circular Blobs: " + str(len(keypoints)) 
# # cv2.putText(blobs, text, (20, 550), 
# #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
  
# # Show blobs 
# cv2.imshow("Filtering Circular Blobs Only", blobs) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 





# # threshold to binary
# thresh = cv2.threshold(dilated_image, 128, 255, cv2.THRESH_BINARY)[1]

# # find contours
# #label_img = img.copy()
# contour_img = dilated_image.copy()
# contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# index = 1
# isolated_count = 0
# cluster_count = 0
# for cntr in contours:
#     area = cv2.contourArea(cntr)
#     convex_hull = cv2.convexHull(cntr)
#     convex_hull_area = cv2.contourArea(convex_hull)
#     ratio = area / convex_hull_area
#     #print(index, area, convex_hull_area, ratio)
#     #x,y,w,h = cv2.boundingRect(cntr)
#     #cv2.putText(label_img, str(index), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
#     if ratio < 0.91:
#         # cluster contours in red
#         cv2.drawContours(contour_img, [cntr], 0, (0,0,150), 10)
#         cluster_count = cluster_count + 1
#     else:
#         # isolated contours in green
#         cv2.drawContours(contour_img, [cntr], 0, (0,150,0), 10)
#         isolated_count = isolated_count + 1
#     index = index + 1
    
# print('number_clusters:',cluster_count)
# print('number_isolated:',isolated_count)

# # save result
# cv2.imwrite("blobs_connected_result.jpg", contour_img)

# # show images
# cv2.imshow("thresh", thresh)
# #cv2.imshow("label_img", label_img)
# cv2.imshow("contour_img", contour_img)
# cv2.waitKey(0)



gray = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
contours, hierarchies = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

blank = np.zeros(thresh.shape[:2], 
                 dtype='uint8')
 
cv2.drawContours(blank, contours, -1, 
                (255, 0, 0), 1)
 
cv2.imwrite("Contours.png", blank)

seeds_pos = np.zeros((0,2), dtype=int)

for i in contours:

    # print(i)
    if [dilated_image.shape[1]-1, dilated_image.shape[0]-1] in i:
        print("Found\n\n")
    else:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(dilated_image, [i], -1, (0, 255, 0), 2)
            cv2.circle(dilated_image, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(dilated_image, "center", (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            seeds_pos = np.vstack((seeds_pos, np.array((cx, cy))))
    # print(f"x: {cx} y: {cy}")

cv2.imshow("contour_img", dilated_image)
cv2.waitKey(0)


print(dilated_image.shape)
print("Seeds Position: ", seeds_pos)
print("Countours:", contours)


# fill_img = np.stack((image1, np.zeros(image1.shape, dtype=np.uint8), np.zeros(image1.shape, dtype=np.uint8)), axis=2, dtype=np.uint8)
# print(fill_img.shape)
# cv2.imshow("image1", fill_img)
# cv2.waitKey(0)

# h,w,_ = fill_img.shape
# seed = seeds_pos[0]

# mask = np.zeros((h+2,w+2),np.uint8)

# floodflags = 4
# floodflags |= cv2.FLOODFILL_MASK_ONLY
# floodflags |= (255 << 8)

# num,fill_img,mask,rect = cv2.floodFill(fill_img, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
# print(num,fill_img,mask,rect)

# # cv2.imwrite("seagull_flood.png", mask)
# cv2.imshow("fill", mask)
# cv2.waitKey(0)






# Load an image (replace 'your_image.jpg' with the actual file path)
image = np.stack((image1, np.zeros(image1.shape, dtype=np.uint8), np.zeros(image1.shape, dtype=np.uint8)), axis=2, dtype=np.uint8)
image_blurred = cv2.GaussianBlur(image, (11, 11), 0)
_,image = cv2.threshold(image_blurred, 50, 255, cv2.THRESH_BINARY)

# Create an empty mask with an extra channel for the alpha channel
mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)



# Set the fill color (replace with the desired BGR color)
fill_color = (0, 0, 255)

# Set the lower and upper difference thresholds for flood-fill
lower_diff = (10, 10, 10)
upper_diff = (10, 10, 10)

result = image.copy()

for seed_point in seeds_pos:
    # # Set the seed point (replace with the coordinates of your desired seed point)
    # seed_point = seeds_pos[0]

    # Perform flood-fill
    cv2.floodFill(image, mask, seed_point, fill_color, lower_diff, upper_diff, cv2.FLOODFILL_MASK_ONLY)

    # Draw a circle to represent the seed point on the original image
    cv2.circle(image, seed_point, 3, (0, 255, 0), -1)

    # Extract the filled mask
    filled_mask = mask[1:-1, 1:-1]

    # Overlay the filled mask on the original image
    result[filled_mask > 0] = fill_color

# Display the original image, seed point, and filled mask
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.imshow('Seed Point and Filled Mask', result)
cv2.waitKey(0)

