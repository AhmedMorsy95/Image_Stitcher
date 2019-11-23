import numpy as np
import cv2


class stitcher:
    def __init__(self):
        self.first_set_of_points = []
        self.second_set_of_points = []

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

        h1 = np.linalg.lstsq(a, b, rcond=None)[0]
        h1 = np.insert(h1, [8], [1.0])
        h1 = h1.reshape(3,3)
        h2, status = cv2.findHomography(np.array(self.first_set_of_points), np.array(self.second_set_of_points))
        print('Our Matrix\n',h1, '\nOpenCV matrix\n', h2)
        return h1
