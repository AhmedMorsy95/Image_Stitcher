from stitcher import stitcher
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


if __name__ == '__main__':

    image_stitcher = stitcher(mpimg.imread('../1.jpg'),mpimg.imread('../2.jpg'))
    #image_stitcher.manually_collect_points(5)
    #image_stitcher.detect_keypoints_and_features()
    image_stitcher.calculate_homography_matrix()

    # draw the 2 images after being wraped using the calculated homography matrix
    # warp = cv2.warpPerspective(image_stitcher.image1,image_stitcher.h,(640,480))
    # plt.figure(1)
    # plt.imshow(warp)
    # warp2 = cv2.warpPerspective(image_stitcher.image2, image_stitcher.h, (640, 480))
    # plt.figure(2)
    # plt.imshow(warp2)
    # plt.show()