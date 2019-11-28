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
        self.matches = None

    def manually_collect_points(self,k):
        plt.imshow(self.image1)
        self.first_set_of_points = plt.ginput(k)
        plt.imshow(self.image2)
        self.second_set_of_points = plt.ginput(k)

    def calculate_homography_matrix(self):
        # this function calculates the matrix using points calculated previously
        # 2 methods are provided implemented method and built-in

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
        h2, status = cv2.findHomography(np.array(self.first_set_of_points), np.array(self.second_set_of_points), cv2.RANSAC)
        print('Our Matrix\n',h1, '\nOpenCV matrix\n', h2)
        self.h = h1


    def detect_keypoints_and_features(self):
        # instead of manually choosing some matching points, we will use opencv's method
        descriptor = cv2.xfeatures2d.SIFT_create()

        # detect keypoints and extracts local invariant descriptors
        keypoints1, features1 = descriptor.detectAndCompute(self.image1, None)
        keypoints2, features2 = descriptor.detectAndCompute(self.image2, None)

        # convert to float
        keypoints1 = np.float32([k.pt for k in keypoints1])
        keypoints2 = np.float32([k.pt for k in keypoints2])

        # match the points
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(features1, features2, 2)

        # apply Lowe’s ratio test, which is used to determine high-quality feature matches.
        # Typical values for Lowe’s ratio are normally in the range [0.7, 0.8].
        # this helps in false-positive match pruning.

        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good.append((m[0].trainIdx, m[0].queryIdx))

        self.first_set_of_points = np.float32([keypoints1[i] for (_, i) in good])
        self.second_set_of_points = np.float32([keypoints2[i] for (i, _) in good])
        self.showMatches(good,keypoints1,keypoints2)
        self.matches = good

    def showMatches(self, matches, keypoints1, keypoints2):
        # this function draws the 2 pictures and a line between each 2 pixels considered a match

        (hA, wA) = self.image1.shape[:2]
        (hB, wB) = self.image2.shape[:2]

        # construct the new image
        matchesImage = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        matchesImage[0:hA, 0:wA] = self.image1
        matchesImage[0:hB, wA:] = self.image2

        for (trainIdx, queryIdx) in matches:
            ptA = (int(keypoints1[queryIdx][0]), int(keypoints1[queryIdx][1]))
            ptB = (int(keypoints2[trainIdx][0]) + wA, int(keypoints2[trainIdx][1]))
            cv2.line(matchesImage, ptA, ptB, (0, 255, 255), 2)

        plt.imshow(matchesImage)
        plt.show()

    def wrap(self):
        result = cv2.warpPerspective(self.image1, self.h, (self.image1.shape[1] + self.image2.shape[1], self.image1.shape[0]))
        result[0:self.image2.shape[0], 0:self.image2.shape[1]] = self.image2
        plt.imshow(result)
        plt.show()
        
    def warp_image_perspective(self, homography_mat):
        src_h, src_w = self.image1.shape[:2] #640,480
        lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])
        trans_lin_homg_pts = homography_mat.dot(lin_homg_pts)
        trans_lin_homg_pts /= trans_lin_homg_pts[2, :]

        minX = np.min(trans_lin_homg_pts[0, :])
        minY = np.min(trans_lin_homg_pts[1, :])
        maxX = np.max(trans_lin_homg_pts[0, :])
        maxY = np.max(trans_lin_homg_pts[1, :])

        dst_sz = list(self.image2.shape)

        pad_sz = dst_sz.copy()  # to get the same number of channels

        pad_sz[0] = np.round(np.maximum(dst_sz[0], maxY) - np.minimum(0, minY)).astype(int)
        pad_sz[1] = np.round(np.maximum(dst_sz[1], maxX) - np.minimum(0, minX)).astype(int)
        dst_pad = np.zeros(pad_sz, dtype=np.uint8)

        # add translation to the transformation matrix to shift to positive values
        anchorX, anchorY = 0, 0
        transl_transf = np.eye(3, 3)
        if minX < 0:
            anchorX = np.round(-minX).astype(int)
            transl_transf[0, 2] += anchorX
        if minY < 0:
            anchorY = np.round(-minY).astype(int)
            transl_transf[1, 2] += anchorY
        new_transf = transl_transf.dot(homography_mat)
        new_transf /= new_transf[2, 2]

        dst_pad[anchorY:anchorY + dst_sz[0], anchorX:anchorX + dst_sz[1]] = self.image2

        warped = cv2.warpPerspective(self.image1, new_transf, (pad_sz[1], pad_sz[0]))

        return dst_pad, warped
