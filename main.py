from stitcher import stitcher
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


if __name__ == '__main__':

    image_stitcher = stitcher(mpimg.imread('../p1.jpg'), mpimg.imread('../p2.jpg'))
    # uncomment only one of the 2 methods either manually choose points or use builtin method.
    # for the manual points it's required more than 4. However the more, the better.
    # image_stitcher.manually_collect_points(6)
    image_stitcher.detect_keypoints_and_features()
    image_stitcher.calculate_homography_matrix()
    # image_stitcher.verify_homography()
    image_stitcher.wrap()
#     dst_pad, warped = image_stitcher.warp_image_perspective(image_stitcher.h)

    # alpha = 0.5
    # beta = 1 - alpha
    # blended = cv2.addWeighted(warped, alpha, dst_pad, beta, 1.0)
    # cv2.imshow("Blended Warped Image", blended)
    # cv2.waitKey(0)
#     plt.figure(1)
#     plt.imshow(dst_warped)
#     plt.show()
