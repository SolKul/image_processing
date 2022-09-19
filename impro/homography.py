import numpy as np
import cv2

#自作モジュール
from . import ransac
from . import impro

def normalize(points):
    """列方向を一つの座標とみなし、最後の行が1になるように、最後の行で割り、正規化する"""
    if len(points.shape) != 2:
        raise ValueError("points dimension should be 2")
    # ブロードキャストが分かりやすいように、1×列数の2次元行列で割る
    return points/points[-1:,:]

def make_homog(points):
    """
    同次座標系にする。もともとの座標行列と1を並べた行とを縦に結合する。
    """
    if len(points.shape) != 2:
        raise ValueError("points dimension should be 2")
    if points.shape[0] != 2:
        raise ValueError("points should be expressed in xy coods")
    return np.vstack(
        (
            points,
            np.ones((1,points.shape[1]))
        )
    )

def H_from_points(fp:np.ndarray,tp:np.ndarray):
    """ 線形なDLT法を使って fpをtpに対応づけるホモグラフィー行列Hを求める。
    点は自動的に調整される """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    if fp.shape[0] != 3:
        raise RuntimeError("points should be expressed in homography coods")
    # 点を調整する（数値計算上重要）
    # 開始点
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    c_1:np.ndarray = np.diag([1/maxstd, 1/maxstd, 1])
    c_1[0][2] = -m[0]/maxstd
    c_1[1][2] = -m[1]/maxstd
    fp = c_1 @ fp
    # 対応点
    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9
    c_2:np.ndarray = np.diag([1/maxstd, 1/maxstd, 1])
    c_2[0][2] = -m[0]/maxstd
    c_2[1][2] = -m[1]/maxstd
    tp = c_2 @ tp
    # 線形法のための行列を作る。対応ごとに2つの行になる。
    nbr_correspondences = fp.shape[1]
    mat_A = np.zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):
        mat_A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
            tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        mat_A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
            tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
    U,S,V = np.linalg.svd(mat_A)
    H = V[8].reshape((3,3))

    # 調整を元に戻す
    H = np.dot(np.linalg.inv(c_2),H @ c_1)
    # 正規化して返す
    return H / H[2,2]

# -

def compute_rasac_homology(img_query_orig, img_train_orig, MIN_MATCH_COUNT=10, show_detail=False, save_result=False):
    """
    query画像とtrain画像についてakazeでマッチングし、
    ransacによって外れ値を除去してHomology行列を算出する。

    Args:
        MIN_MATCH_COUNT (int):mathesの数の最小値。これ以下だとHomologyを計算しない
    """
    img_query = img_query_orig.copy()
    img_train = img_train_orig.copy()

    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()

    # key pointとdescriptorを計算
    kp1, des1 = akaze.detectAndCompute(img_query, None)
    kp2, des2 = akaze.detectAndCompute(img_train, None)

    # matcherとしてflannを使用。
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)

    # ANNで近傍２位までを出力
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    # 2番めに近かったkey pointと差があるものをいいkey pointとする。
    good_matches = []
    for i in range(len(matches)):
        if(len(matches[i])<2):
            continue
        m,n=matches[i]
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    # descriptorの距離が近かったもの順に並び替え
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    if(show_detail):
        # 結果を描写
        img_result = cv2.drawMatches(
            img_query, kp1, img_train, kp2, good_matches[:10], None, flags=2)
        ip.show_img(img_result, figsize=(20, 30))
        print('queryのkp:{}個、trainのkp:{}個、good matchesは:{}個'.format(
            len(kp1), len(kp2), len(good_matches)))

    # ransacによって外れ値を除去してHomology行列を算出する。
    # opencvの座標は3次元のarrayで表さなければならないのに注意

    if len(good_matches) > MIN_MATCH_COUNT:
        # matching点の座標を取り出す
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # ransacによって外れ値を除去
        homology_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, 5.0)

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None
        return None, None

    if(show_detail or save_result):
        # 結果を描写
        matchesMask = mask.ravel().tolist()

        # query画像の高さ、幅を取得し、query画像を囲う長方形の座標を取得し、
        # それを算出された変換行列homology_matrixで変換する
        # 変換した長方形をtrain画像に描写
        h, w = img_query.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, homology_matrix)
        cv2.polylines(img_train, [np.int32(dst)],
                      True, (255, 100, 0), 3, cv2.LINE_AA)

        num_draw = 50

        draw_params = dict(
            #     matchColor = (0,255,0), # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask[:num_draw],  # draw only inliers
            flags=2)

        img_result_2 = cv2.drawMatches(
            img_query, kp1, img_train, kp2, good_matches[:num_draw], None, **draw_params)

    if(show_detail):
        ip.show_img(img_result_2, figsize=(20, 30))
        num_inlier = (mask == 1).sum()
        print('inlier:%d個' % num_inlier)
    if(save_result):
        ip.imwrite('ransac_match.jpg', img_result_2)

    return homology_matrix, mask

# +
# import numpy as np
# from matplotlib import pyplot as plt

# from sklearn import linear_model, datasets


# n_samples = 1000
# n_outliers = 50


# X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
#                                       n_informative=1, noise=10,
#                                       coef=True, random_state=0)

# # Add outlier data
# np.random.seed(0)
# X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
# y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# # Fit line using all data
# lr = linear_model.LinearRegression()
# lr.fit(X, y)

# # Robustly fit linear model with RANSAC algorithm
# ransac = linear_model.RANSACRegressor()
# ransac.fit(X, y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# # Predict data of estimated models
# line_X = np.arange(X.min(), X.max())[:, np.newaxis]
# line_y = lr.predict(line_X)
# line_y_ransac = ransac.predict(line_X)

# # Compare estimated coefficients
# print("Estimated coefficients (true, linear regression, RANSAC):")
# print(coef, lr.coef_, ransac.estimator_.coef_)

# lw = 2
# plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
#             label='Inliers')
# plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
#             label='Outliers')
# plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
# plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
#          label='RANSAC regressor')
# plt.legend(loc='lower right')
# plt.xlabel("Input")
# plt.ylabel("Response")
# plt.show()

# +


# -

class RansacModel:
    """ http://www.scipy.org/Cookbook/RANSAC のransac.pyを用いて
    ホモグラフィーを当てはめるためのクラス """
    def __init__(self,debug=False):
        self.debug = debug
        
    def fit(self, data):
        """ 4つの対応点にホモグラフィーを当てはめる """
        # H_from_points() を当てはめるために転置する
        data = data.T
        # 元の点
        fp = data[:3,:4]
        # 対応点
        tp = data[3:,:4]
        # ホモグラフィーを当てはめて返す
        return H_from_points(fp,tp)
    
    def get_error( self, data, H):
        """ すべての対応にホモグラフィーを当てはめ、各変換点との誤差を返す。"""
        data = data.T
        # 元の点
        fp = data[:3]
        # 対応点
        tp = data[3:]
        # fpを変換
        fp_transformed = H @ fp
        # 同次座標を正規化
        nz = np.nonzero(fp_transformed[2])
        for i in range(3):
            fp_transformed[i][nz] = fp_transformed[i][nz]/fp_transformed[2][nz]
            
        # 1点あたりの誤差を返す
        return np.sqrt( np.sum((tp-fp_transformed)**2,axis=0) )


def H_from_ransac(fp,tp,model,maxiter=1000,match_threshold=10):
    """ RANSACを用いて対応点からホモグラフィー行列Hをロバストに推定する
    (ransac.py は http://www.scipy.org/Cookbook/RANSAC を使用)
    入力: fp,tp (3*n 配列) 同次座標での点群 """
    # 対応点をグループ化する
    data = np.vstack((fp,tp))
    # Hを計算して返す
    H,ransac_data = ransac.ransac(data.T,model,4,maxiter,match_threshold,10,
                                  return_all=True)
    return H,ransac_data['inliers']
