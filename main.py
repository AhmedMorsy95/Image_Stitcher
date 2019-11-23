from stitcher import stitcher
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



if __name__ == '__main__':
    image_stitcher = stitcher()

    image1 = mpimg.imread('../1.jpg')
    plt.imshow(image1)
    image_stitcher.first_set_of_points = plt.ginput(4)

    image2 = mpimg.imread('../2.jpg')
    plt.imshow(image2)
    image_stitcher.second_set_of_points = plt.ginput(4)

    image_stitcher.calculate_homography_matrix()
