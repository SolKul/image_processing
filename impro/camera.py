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

# +
import numpy as np
from scipy import linalg

class Camera:
    '''
    カメラ行列Pを与えてカメラオブジェクトを作るモデル。
    
    Args:
        P (array):カメラ行列P=K[R|t]。
    '''
    def __init__(self,P:np.ndarray):
        self.P=P
        self.K=None 
        self.R=None
        self.t=None
        self.c=None

    def project(self,X):
        x=self.P @ X
        x=x/x[2]
        return x   

    def factor(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """P=K[R|t]。に従い、カメラ行列をK,R,tに分解する

        Returns:
            tuple[np.ndarray,np.ndarray,np.ndarray]: _description_
        """
        # 最初の3x3の部分を分解する
        mtx_k,mtx_r=linalg.rq(self.P[:,:3])
        # Kの対角成分が正になるようにする
        mtx_t=np.diag(np.sign(np.diag(mtx_k)))
        # 以下2行の条件分岐は不要では？
        # if np.linalg.det(mtx_t) < 0:
        #     mtx_t[1,1] *= -1
        #K,Rの行列式を正にする
        self.K= np.dot(mtx_k,mtx_t)
        self.R= np.dot(mtx_t,mtx_r)
        self.t= np.linalg.inv(self.K) @ self.P[:,3]
        
        return self.K, self.R, self.t

    def factor_solem(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """P=K[R|t]。に従い、カメラ行列をK,R,tに分解する、教科書通りのじ実装
        不要な条件分岐がある

        Returns:
            tuple[np.ndarray,np.ndarray,np.ndarray]: _description_
        """
        # 最初の3x3の部分を分解する
        mtx_k,mtx_r=linalg.rq(self.P[:,:3])
        # Kの対角成分が正になるようにする
        mtx_t=np.diag(np.sign(np.diag(mtx_k)))
        # 以下2行の条件分岐は不要では？
        if np.linalg.det(mtx_t) < 0:
            mtx_t[1,1] *= -1
        #K,Rの行列式を正にする
        self.K= np.dot(mtx_k,mtx_t)
        self.R= np.dot(mtx_t,mtx_r)
        self.t= np.linalg.inv(self.K) @ self.P[:,3]
        
        return self.K, self.R, self.t
    
    def center(self):
        """カメラ中心を計算して返す"""
        if self.c is not None:
            return self.c
        else:
            self.factor()
            self.c= - self.R.T @ self.t
            return self.c
            
    
def rotation_matrix(a):
    """ ベクトルaを軸に回転する3Dの回転行列を返す """
    R = np.eye(4)
    R[:3,:3] = linalg.expm(
        [
            [0    ,-a[2], a[1]],
            [a[2] ,0    ,-a[0]],
            [-a[1],a[0] ,0    ]
        ])
    return R

def calculate_camera_matrix_w_sz(sz,sz_orig=(6000,4000),lens='PZ',f_orig=(4188,4186)):
    """
    異なる解像度でのカメラの内部パラメータKを計算する関数
    Args:
        sz (int):扱う画像サイズ。もともとの画像から縮小していた場合など。
        sz_orig (int):(6000,4000)はα6000で24MPで撮影したときの解像度
        lens (str): 'PZ'か'SEL'
        f_orig (int):(5266,5150)はα6000でSEL18200で焦点距離18のときの焦点距離
        (4188,4186)はPZ1650でズーム16のときの焦点距離
    """
    if(lens=='PZ'):
        f_orig=(4188,4186)
    elif(lens=='SEL'):
        f_orig=(5266,5150)
    fx_orig,fy_orig=f_orig
    width,height=sz
    width_orig,height_orig=sz_orig
    fx=fx_orig * width /width_orig
    fy=fy_orig * height /height_orig
    K = np.diag([fx,fy,1])
    K[0,2]=0.5*width
    K[1,2]=0.5*height
    return 