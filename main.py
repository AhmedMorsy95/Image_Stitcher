from stitcher import stitcher
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


if __name__ == '__main__':

    image_stitcher = stitcher(mpimg.imread('../1.jpg'), mpimg.imread('../2.jpg'))
    # uncomment only one of the 2 methods either manually choose points or use builtin method.
    # for the manual points it's required more than 4. However the more, the better.
    # image_stitcher.manually_collect_points(6)
    # image_stitcher.detect_keypoints_and_features()
    image_stitcher.calculate_homography_matrix()
#     image_stitcher.wrap()
    dst_pad, warped = image_stitcher.warp_image_perspective(homography_mat)

    alpha = 0.5
    beta = 1 - alpha
    blended = cv2.addWeighted(warped, alpha, dst_pad, beta, 1.0)
    plt.imshow(blended)
    plt.show()
