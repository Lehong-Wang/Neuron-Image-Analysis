
import cv2
import numpy as np



CHANNEL_NUM = 4
BINARY_THRESHOLD = 150


def read_tif(tif_file, n_channels=CHANNEL_NUM, plot=False):
    """Read a TIFF file and parse into a matrix of images.
        
        TIFF images are structured as channels, layers, and images.
        Ploting all channels is somehow broken
        Returns:
            4D matrix of images with dimension: (n_channels, n_layers, image(2D))
    
    """
    # returns a tuple of channels of images 
    _, multi_channel_data = cv2.imreadmulti(tif_file, [], cv2.IMREAD_ANYCOLOR)

    n_layers = len(multi_channel_data) // n_channels
    img_height, img_width, _ = multi_channel_data[0].shape
    # image must be of dtype np.uint8 for opencv to recognize it correctly
    # especially when the image is created from np array
    all_img = np.zeros((n_channels, n_layers, img_height, img_width), dtype=np.uint8)

    for ii, this_data in enumerate(multi_channel_data):
        # the image data is stored in the third channel of this_data, don't know why
        all_img[ii%n_channels, ii//n_channels] = this_data[:,:,2]
    
    print("All_Imgae Shape:", all_img.shape)

    if plot is True:
        cv2.imshow("Channel 0, Layer 0", multi_channel_data[0][:,:,2])
        cv2.waitKey(0)
        cv2.imshow("Channel 1, Layer 0", multi_channel_data[1][:,:,2])
        cv2.waitKey(0)

        canvas = np.zeros((n_channels*img_height, n_layers*img_width), dtype=np.uint8)
        for row in range(n_channels):
            for col in range(n_layers):
                plot_img = all_img[row, col]
                canvas[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = plot_img
        # original image too big to display
        small = cv2.resize(canvas, (n_channels * img_height // 10, n_layers * img_width // 10))
        cv2.imshow("All IMG", small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return all_img




def get_image_overlap_position(img1, img2, binary_threshold=BINARY_THRESHOLD, plot=False):
    """Find the exact position of overlap between two images.

        Two images should be same size, and single channel (2D)
            because contour extraction only supports single channel
        This function should not affect the number of channels

        Returns:
            nx2 array of position of the centor of overlapping regions
    
    """
    # print(f"Image1: {img1.shape}\tImage2: {img2.shape}")

    if img1.shape[0] != img2.shape[0] or len(img1.shape) != 2:
        print(f"ERROR: Given image have wrong shape.\nImage1: {img1.shape}\tImage2: {img2.shape}")
        return np.zeros((0,2))

    _, binary_img1 = cv2.threshold(img1, binary_threshold, 255, cv2.THRESH_BINARY)
    _, binary_img2 = cv2.threshold(img2, binary_threshold, 255, cv2.THRESH_BINARY)
    # apply binary AND operator
    result = cv2.bitwise_and(binary_img1, binary_img2)
    print("result", result.shape)

    kernel5 = np.ones((5, 5), np.uint8)
    kernel11 = np.ones((11, 11), np.uint8)
    # clean the result
    dilated_img = cv2.dilate(result, kernel5, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel11, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel11, iterations=1)

    print("Dilation Result", dilated_img.shape)

    contour_img = dilated_img.copy()

    # get contours of overlapping regions
    contours, hierarchies = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # convert to 3 channel for ploting with colors
    contour_img = np.repeat(contour_img[:, :, np.newaxis], 3, axis=2)
    # print("Contour shape", contour_img.shape)

    # find centor of regions
    seeds_pos = np.zeros((0,2), dtype=int)

    for this_contour in contours:
        # remove the boundary of image detected as contour
        if [contour_img.shape[1]-1, contour_img.shape[0]-1] in this_contour:
            # print("Found\n\n")
            continue
        M = cv2.moments(this_contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(contour_img, [this_contour], -1, (0, 255, 0), 2)
            cv2.circle(contour_img, (cx, cy), 3, (0, 0, 255), -1)
            # cv2.putText(dilated_img, "center", (cx - 20, cy - 20),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            seeds_pos = np.vstack((seeds_pos, np.array((cx, cy))))
            # print(f"x: {cx} y: {cy}")


    # print(dilated_img.shape)
    # print("Seeds Position: ", seeds_pos)
    # print("Countours:", contours)


    if plot is True:
        cv2.imshow('Binary Image 1', binary_img1)
        cv2.waitKey(0)
        cv2.imshow('Binary Image 2', binary_img2)
        cv2.waitKey(0)
        cv2.imshow('Overlap', result)
        cv2.waitKey(0)
        cv2.imshow('Eroded Image', eroded_img)
        cv2.waitKey(0)
        cv2.imshow('Dilated Image', dilated_img)
        cv2.waitKey(0)
        cv2.imshow("Contour Image", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return seeds_pos


def get_regions_from_seeds(img, seeds_pos, plot=False):
    img_blurred = cv2.GaussianBlur(img, (11, 11), 0)
    _,img_binary = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY)

    # mask for floodfill should be 2 pixels larger than image
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

    fill_color = (0, 0, 255)
    lower_diff = (10, 10, 10)
    upper_diff = (10, 10, 10)

    for seed_point in seeds_pos:
        cv2.floodFill(img_binary, mask, seed_point, fill_color, lower_diff, upper_diff, cv2.FLOODFILL_MASK_ONLY)
    # crop the extra pixels
    region_mask = mask[1:-1, 1:-1]

    if plot is True:
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)
        cv2.imshow('Filtered Image', img_binary)
        cv2.waitKey(0)

        masked_img = img.copy()
        # make 3 channels for color
        masked_img = np.stack((masked_img, np.zeros(masked_img.shape, dtype=np.uint8), np.zeros(masked_img.shape, dtype=np.uint8)), axis=2, dtype=np.uint8)
        doted_img = masked_img.copy()

        masked_img[region_mask > 0] = fill_color
        for seed_point in seeds_pos:
            cv2.circle(doted_img, seed_point, 3, (0, 255, 0), -1)

        cv2.imshow('Seed Position', doted_img)
        cv2.waitKey(0)
        cv2.imshow('Masked Image', masked_img)
        cv2.waitKey(0)

    return region_mask



if __name__ == '__main__':
    tif_file = "img_crop.tif"
    all_img = read_tif(tif_file, plot=False)
    # img1 = np.repeat(all_img[0,0][:, :, np.newaxis], 3, axis=2)
    # img2 = np.repeat(all_img[1,0][:, :, np.newaxis], 3, axis=2)
    seeds_pos = get_image_overlap_position(all_img[0,0], all_img[1,0], plot=True)
    # get_image_overlap_position(img1, img2, plot=True)

    region_mask = get_regions_from_seeds(all_img[0,0], seeds_pos, plot=True)


