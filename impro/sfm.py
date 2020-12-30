# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# あるカメラについて、ワールド座標系の3Dの点Xと画像上の点$x$の変換が$\lambda x=PX$で表せられたとします。  
# 点Xを任意の（4×4の）ホモグラフィー行列 H を使って別の座標系での点、$\hat{X}=HX$に座標変換したとします。  
# このとき、別のワールド座標系でのP行列は$\hat{P}=PH^{－1}$となります。
# $$
# \lambda x=PX=PH^{-1}HX=\hat{P}\hat{X}
# $$
#   
# ![](https://cdn-ak.f.st-hatena.com/images/fotolife/Z/Zellij/20120823/20120823234831.png)
#   
# ## エピポーラ線
#
# ・直線 OL-Xは、左のカメラでは1つの点に投影される。
# ・右のカメラの直線 eR - xRはエピポーラ線（epipolar line）と呼ばれる。
# ・エピポーラ線は点Xの3次元空間位置によって一意に定まるが、すべてのエピポーラ線はエピポーラ点（図ではeL、eR）を通る。
# ・また逆に、エピポーラ点を通る直線は、すべてエピポーラ線であると言える。
#
#
# ## エピポーラ面
# ・点X、OL、ORの3点を通る平面はエピポーラ面と呼ばれる。
# ・エピポーラ面と、投影面の交線はエピポーラ線と一致する。（エピポーラ線上にはエピポーラ点がある。）
#
# ## エピポーラ制約
# 2つのカメラの互いの位置関係が既知であれば、次のことがいえる。
# ・点Xの左カメラでの投影xLが与えられると、右カメラのエピポーラ線 eR - xRが定義される。そして、点Xの右カメラでの投影xRは、このエピポーラ線上のどこかにある。これをエピポーラ制約（epipolar constraint）と呼ぶ。
# ・つまり、2つのカメラで同じ点を捕捉しているとした場合、必ずそれは互いのエピポーラ線上に乗る。もしエピポーラ線に乗っていないのであれば、それは同じ点を見ていない（対応付けが正しくない）ということになる。
# ・また、一方のカメラで見ている点が、他方のカメラのどこに映っているだろうか、という問題に対しては、エピポーラ線上を調べれば十分であることがわかる。
# ・もし、対応付けが正しくて、xLとxRの位置がわかっているのであれば、点Xの3次元空間での位置を決定することができる（三角法による位置の決定）。  
# エピポーラ制約は
# $$
# x_2^TFx_1=0
# $$
# と表すことができる。これは$l_1=Fx_1$としたとき、$x_2^Tl_1=0$は、$x_2$がエピポーラ線上にあることを表す。  
#
# そして$F$は基礎行列と呼ばれる。
# $$
# F=K_2^{-T}S_tRK_1^{-1}
# $$
# また$E=S_tR$は基本行列と呼ばれる
#
# ## エピポール、エピ極
# 複数の三次元上の点のそれぞれについて、画像上にエピポーラ線があります。それら複数のエピポーラ線は、画像上で必ず１点で交わります。それがエピポール、エピ極です。この点は、他方の画像のカメラ中心を自分の画像に投影した点です。  
# エピポールはあらゆるエピポーラ線上にあるので、
# $$
# l_1e_2=0,\forall l_1
# $$
# と表せます。
# つまり
# $$
# x_1^TFe_2=0,\forall x_1
# $$
# とどの点に対してもエピポール制約が成り立つということです。つまり
# $$
# Fe_2=0
# $$
# とすることができます。

# ## ８点法について
# 行列の零空間を求める。  
# $$
# Af=0
# $$
# を最小二乗法で解く  
# 普通に解けばf=0が階だがそれがほしいわけではない。  
# $Af=0$は固有値の式$Af=\lambda f$に似ているので  
# $A$の固有値のうち最小の固有値の固有ベクトルを正規化したものを解とする。  
# 当然$A$は正方行列じゃないので、固有値なんてないが、  
# $A=USV$
# と特異値分解して、一番小さい特異値$S[-1]$に対応するVの最後の行のベクトル、$V[-1]$を解とする

# ## 点群からカメラ行列の計算
# ### カメラの内部パラメータKがわからない場合
# ### カメラの内部パラメータKが既知の場合(どちらについても)
# 画像上の点$x_2$は
# $$
# x_2=K(R|t)X_2
# $$
# だから
# $$
# x_{2K}=K^{-1}x_2=K^{-1}K(R|t)X_2=(R|t)X_2
# $$
# と$x_{2K}$はカメラを原点とした3次元上の点となる。
# よって
# $$
# x_2^TFx_1=x_2^TK_2^{-T}S_tRK_1^{-1}x_1=x_{2K}^TS_tRx_{1K}=x_{2K}^TEx_{1K}=0
# $$
# となる。よって基本行列Eは8点法を用いて計算できることになる。  
# #### 流れ
# 前提: カメラの内部パラメータは既知
# 1. 2枚の画像から特徴点を取り出しマッチングする。
# 2. それぞれの画像上の点の座標にカメラの内部パラメータを内積し、$x_{1K}$、$x_{2K}$を求める。
# 3. $x_{2K}^TEx_{1K}=0$を元に8点法でEを求める。

# +
import numpy as np
import matplotlib.pyplot as plt
import os

#自作モジュール
import image_processing as ip
import camera
import ransac
import desc_val as dv


# +
def compute_fundamental(x1,x2):
    """ 
    正規化8点法を使って対応点群(x1,x2:3*nの配列)
    から基礎行列を計算する。各列は次のような並びである。
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] 
    """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # 方程式の行列を作成する
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i]*x2[0, i], x1[0, i]*x2[1, i], x1[0, i]*x2[2, i],
                x1[1, i]*x2[0, i], x1[1, i]*x2[1, i], x1[1, i]*x2[2, i],
                x1[2, i]*x2[0, i], x1[2, i]*x2[1, i], x1[2, i]*x2[2, i]]

    # 線形最小2乗法で計算する
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    #Fの制約
    #最後の特異値を0にして階数2にする。
    U,S,V = np.linalg.svd(F)
    S[2] =0
    F = U @ np.diag(S) @ V
    
    return F

def compute_epipole(F):
    """ 基礎行列Fから（右側）のエピ極を計算する。
    （左のエピ極を計算するには F.Tを用いる """
    #Fの零空間(Fx=0)を返す。
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]


# -

# エピポーラ線を描写することを考える。
# $$
# x_1Fx_2=l_1x_2=0
# $$
# において、
# $l_1=(a_1,a_2,a_3)^T$、$x_2=(x,y,1)^T$とすると
# $$
# a_1x+a_2y+a_3=0
# $$
# で、yについて解くと
# $$
# y=\frac{a_3+a_1x}{-a_2}
# $$

def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    """
    エピ極とエピポーラ線Fx=0を画像に描写する
    Args:
        F (array):基礎行列
        x (array):第2画像上の点
    """
    m,n = im.shape[:2]
    line = F @ x
    
    #エピポーラ線のパラメータと値
    t = np.linspace(0,n,100)
    lt = np.array([(line[2]+line[0]*tt)/-line[1] for tt in t])
    
    #画像中に含まれる線分だけを選ぶ
    ndx = (lt>0) & (lt<m)
    plt.plot(t[ndx],lt[ndx],linewidth=2)
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')


def load_merton_data(data_path='./MertonCollege', show_detail=False):
    """
    https://www.robots.ox.ac.uk/~vgg/data/mview/の"Merton College I"データを読み込む
    Args:
        data_path (str):データを展開したディレクトリのpath。image,2D,3Dがある前提
    Returns:
        corr:ある717点ある3次元上の点が、2D/001.cornes内で示されたのどの2D上の点に対応しているか。
            indexを表す。
    """
    # 画像を読み込む
    im1 = ip.imread(os.path.join(data_path, 'image/001.jpg'))
    im2 = ip.imread(os.path.join(data_path, 'image/002.jpg'))
    if(show_detail):
        ip.show_img(im1)
        ip.show_img(im2)

    # 画像上の2Dの点をリストに読み込む
    points2D = []
    for i in range(3):
        file_path = os.path.join(data_path, '2D/{:0=3}.corners'.format(i+1))
        points2D.append(np.loadtxt(file_path).T)

    # 3Dの点を読み込む
    points3D = np.loadtxt(os.path.join(data_path, '3D/p3d')).T

    # 対応関係を読み込む
    corr = np.genfromtxt(
        os.path.join(data_path, '2D/nview-corners'), dtype='int', missing_values='*')

    # カメラパラメーラをCameraオブジェクトに読み込む
    P = []
    for i in range(3):
        istrct_p = np.loadtxt('MertonCollege/2D/{:0=3}.P'.format(i+1))
        P.append(camera.Camera(istrct_p))

    return im1, im2, points2D, points3D, corr, P


# +
def triangulate_point(x1, x2, P1, P2):
    """最小自乗解を用いて点の組を三角測量する"""

    M = np.zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]

    return X/X[3]


def triangulate(x1, x2, P1, P2):
    """
    x1,x2(3*nの同時座標)の点の2視点三角測量
    """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]

    return np.array(X).T


def compute_P(x, X):
    """
    2D-3Dの対応の組(同時座標系)からカメラ行列Pを計算する
    """

    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match")

    # DLT法で行列を作成する。
    M = np.zeros((3*n, 12+n))
    for i in range(n):
        M[3*i, 0:4] = X[:, i]
        M[3*i+1, 4:8] = X[:, i]
        M[3*i+2, 8:12] = X[:, i]
        M[3*i:3*i+3, i+12] = -x[:, i]

    U, S, V = np.linalg.svd(M)

    return V[-1, :12].reshape((3, 4))


def compute_P_from_fundamental(F):
    """
    第2のカメラ行列(第一のカメラ行列について、P1 = [I 0]を仮定)を、
    基礎行列から計算する。
    """

    e = compute_epipole(F.T)  # left epipole
    Te = skew(e)
    return np.vstack((Te @ F.T).T, e).T


def skew(a):
    """
    任意のvについて、aとvの外積との間に、a x v =Avになる交代行列A
    """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def compute_P_from_essential(E):
    """
    基本行列から第2のカメラ行列を計算する()
    出力は可能性のある4つのカメラ行列
    """

    # Eの階数が2になるようにする。
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(U @ V) < 0:
        V = -V
    E = U @ np.diag([1, 1, 0]) @ V

    # 行列を作成
    Z = skew([0, 0, -1])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # 4つの解を返す。
    P2 = [np.vstack(((U @ W @ V).T, U[:, 2])).T,
          np.vstack(((U @ W @ V).T, -U[:, 2])).T,
          np.vstack(((U @ W.T @ V).T, U[:, 2])).T,
          np.vstack(((U @ W.T @ V).T, -U[:, 2])).T]

    return P2


# +
class RansacModel:
    """
    ransac.pyを用いて、基礎行列を当てはめるためのクラス
    """

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """
        8つの選択した対応を使って基礎行列を推定する。
        """

        # データを転地し、2つの点群に分ける。
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        # 基礎行列を推定して返す。
        F = compute_fundamental_normalized(x1, x2)
        return F

    def get_error(self, data, F):
        """
        すべての対応について、x^T F xを計算し、
        変換された点の誤差を返す。
        """

        # データを転地し、2つの点群に分ける。
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        # 誤差尺度としてSampson距離を使う。
        Fx1 = F @ x1
        Fx2 = F @ x2
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = (np.diag(x1.T @ F @ x2))**2/denom

        # 1点あたりの誤差を返す。
        return err


def compute_fundamental_normalized(x1, x2):
    """
    正規化8点法を用いて対応点群(x1,x2:3*nの配列)
    から基礎行列を計算する。各列は次のような並びである。
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] 
    """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match")

    # 画像の座標を正規化する。
    x1 = x1/x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1*mean_1[0]],
                   [0, S1, -S1*mean_1[1]],
                   [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2/x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2*mean_2[0]],
                   [0, S2, -S2*mean_2[1]],
                   [0, 0, 1]])
    x2 = T2 @ x2

    # 正規化した座標でFを計算する
    F = compute_fundamental(x1, x2)

    # 正規化を元に戻す。
    F = T1.T @ F @ T2

    return F / F[2, 2]


def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    """
    ransacを使って点の対応から基礎行列Fをロバスト推定する。
    Args:
        x1 (array):(3*n配列)同次座標系の点群
    """

    data = np.vstack((x1, x2))

    # Fを計算し、インライアのインデックスと一緒に返す。
    F, ransac_data = ransac.ransac(
        data.T, model, n=8, k=maxiter, t=match_threshold, d=20, return_all=True)
    return F, ransac_data['inliers']
