import numpy as np
import cv2
import matplotlib.pyplot as plt

class stitcher:

    def __init__(self, image1, image2):
        self.first_set_of_points = None
        self.second_set_of_points = None
        self.h = None
        self.image1 = image1
        self.image2 = image2

    def manually_collect_points(self,k):
        plt.imshow(self.image1)
        self.first_set_of_points = plt.ginput(k)
        plt.imshow(self.image2)
        self.second_set_of_points = plt.ginput(k)

    def calculate_homography_matrix(self):
        n = len(self.first_set_of_points)
        if n < 4:
            print('can not calculate need at least 4 points')
            return

        # link for equations : https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
        a = np.zeros((2*n,8))
        b = np.zeros((2*n,1))

        for i in range(0,n):
            current_point = self.first_set_of_points[i]
            current_point2 = self.second_set_of_points[i]

            a[i*2] = [current_point[0], current_point[1], 1, 0, 0, 0, -current_point[0]*current_point2[0], -current_point[1]*current_point2[0]]
            a[i*2+1] = [0, 0, 0, current_point[0], current_point[1], 1, -current_point[0]*current_point2[1], -current_point[1]*current_point2[1]]

            b[i*2] = current_point2[0]
            b[i*2+1] = current_point2[1]

        # h1 is manually calculated homography
        # h2 is opencv's homography

        h1 = np.linalg.lstsq(a, b, rcond=None)[0]
        h1 = np.insert(h1, [8], [1.0])
        h1 = h1.reshape(3,3)
        h2, status = cv2.findHomography(np.array(self.first_set_of_points), np.array(self.second_set_of_points))
        print('Our Matrix\n',h1, '\nOpenCV matrix\n', h2)
        self.h = h2
        return h1


    def detect_keypoints_and_features(self):
        # instead of manually choosing some matching points, we will use opencv's method
        descriptor = cv2.xfeatures2d.SIFT_create()

        # detect keypoints and extracts local invariant descriptors
        keypoints1, features1 = descriptor.detectAndCompute(self.image1, None)
        keypoints2, features2 = descriptor.detectAndCompute(self.image2, None)

        keypoints1 = np.float32([k.pt for k in keypoints1])
        keypoints2 = np.float32([k.pt for k in keypoints2])

        #match the points
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(features1, features2, 2)

        good = []
        # apply Lowe’s ratio test, which is used to determine high-quality feature matches. Typical values for Lowe’s ratio are normally in the range [0.7, 0.8].
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good.append((m[0].trainIdx, m[0].queryIdx))

        self.first_set_of_points = np.float32([keypoints1[i] for (_, i) in good])
        self.second_set_of_points = np.float32([keypoints2[i] for (i, _) in good])
