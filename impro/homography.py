from typing import Union
import numpy as np
import cv2

#自作モジュール
from . import ransac,impro,detection

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

def compute_rasac_homography(
    kp_query:tuple[cv2.KeyPoint,...],
    kp_train:tuple[cv2.KeyPoint,...],
    matches:list[cv2.DMatch],
)->tuple[np.ndarray,np.ndarray]:
    # matching点の座標を取り出す
    src_pts = np.float32(
        [kp_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_train[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # ransacによって外れ値を除去し、ホモグラフィー行列を計算
    # opencvの座標は3次元のarrayで表さなければならないのに注意
    homology_matrix, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homology_matrix,mask

def compute_draw_homography(
    img_query:np.ndarray,
    img_train:np.ndarray,
    min_match_count:int=10,
)->Union[np.ndarray,None]:
    """
    query画像とtrain画像についてakazeでマッチングし、
    ransacによって外れ値を除去してHomology行列を算出する。

    Args:
        min_match_count (int):mathesの数の最小値。これ以下だとHomologyを計算しない
    """
    # 特徴量マッチングによりkpとｍatchesを取得
    kp_query,kp_train,matches=detection.akaze_matching(img_query,img_train)
    # もしmatchesがmin_match_count以下しかなければホモグラフィーは計算しない
    if len(matches) < min_match_count:
        return
    # ransacによって外れ値を除去してHomology行列を算出する。
    homology_matrix, mask =compute_rasac_homography(kp_query,kp_train,matches)

    matchesMask = mask.ravel().tolist()

    # img_queryをimg_train上に移動したとき、どこに移動するか分かりやすくするため、
    # img_train上に長方形を描く
    h,w = img_query.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,homology_matrix)
    img_train_poly=img_train.copy()
    img_train_poly = cv2.polylines(img_train_poly,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # img_queryと長方形を描いたimg_traingを並べ、マッチング結果を表示する
    draw_params = dict(
        matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 2
    )
    img_result = cv2.drawMatches(
        img_query,
        kp_query,
        img_train_poly,
        kp_train,
        matches,
        None,
        **draw_params
    )

    return img_result
