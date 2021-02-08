import numpy as np
import cv2
import os

MIN_MATCH_COUNT = 10
img1 = cv2.imread('query.jpg', 0)  # queryImage

path_dir = r'C:\Users\user\Desktop\AI-based-Dementia-Severity-Prediction\dataset\test2_png\before'
os.chdir(path_dir)
file_list = os.listdir()

for files in file_list :

    img2 = cv2.imread(files, 0)  # trainImage
    print(img1)
    print(img2)
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good.append(m)

    print("Match Points :", len(good), "(> 10)")

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # 물체를 검출하고 왜곡된 박스 형태로 행렬을 반환함
        box = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 255), 2)

        # 검출한 물체 박스의 네 꼭짓점의 좌표를 계산
        x_list = [np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0]]
        x_list.sort()
        y_list = [np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1]]
        y_list.sort()

        #
        cv2.rectangle(img2, (x_list[0], y_list[0]), (x_list[3], y_list[3]), (255, 255, 255), -1)

    else:
        # Not enough matches ERROR
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    cv2.imwrite('test2_png/%s.png' %files, img2)
    cv2.imshow('img', img3)
    cv2.imshow('result', img2)
    cv2.waitKey()
    cv2.destroyAllWindows()