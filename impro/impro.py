from typing import Union
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def imshow(
        img:np.ndarray,
        figsize:tuple[int,int]=(9,6),
        isBGR:bool=True,
        show_mode:str="as is",
        show_axis:bool=False,
        return_fig:bool=False) -> Union[None,plt.Figure]:
    """
    np.array形式の画像を表示する。カラーでもグレースケールでも対応。
    グレースケールの場合、そのまま、スケール、ログスケールで表示するかが選べる。
    show_modeが"as is"の場合そのまま、"sacale"の場合最小値が0、最大値が255になるようスケールして
    "log" :最小値が0、最大値が255になるようログでスケールして表示する。

    Args:
        img (np.ndarray): 表示したい画像
        figsize (tuple[int,int], optional): pltのfigsize. Defaults to (9,6).
        isBGR (bool, optional): opencvで読み込んだ場合、チャネルがBGRなので変換する必要がある. Defaults to True.
        show_mode (str, optional): 上の説明を参照. Defaults to "scale".
        show_axis (bool, optional): x軸、y軸の表示を消す. Defaults to False.
        return_fig (bool, optional): plt.figureで生成したfigを返すか. Defaults to False.

    Raises:
        ValueError: 画像の次元数が2か3でないの場合

    Returns:
        Union[None,plt.Figure]: figを返す場合、ax=fig.axes[0]とすれば、その後表示を付け足せる。
    """
    # ここで生成したfigがcurrent figureとなるので、
    # notebookでこのまま抜けるとこのfigが描写される
    fig=plt.figure(figsize=figsize)
    dim_num= len(img.shape)
    # 画像の次元数は2(グレースケール)か3(カラー)のどちらかに対応
    if dim_num < 2 or dim_num>3:
        raise ValueError("dimenstion sholud be 2 or 3")
    if dim_num==3:
        img=img.astype(np.uint8)
        if isBGR:
            img_cvt=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img_cvt=img
        plt.imshow(img_cvt)
    else:
        if show_mode == "as is":
            img_show=img.astype(np.uint8)
            plt.imshow(img_show,cmap='gray',vmin=0,vmax=255)
        elif show_mode == "scale":
            min_intens=img.min()
            max_intens=img.max()
            img_show=((img-min_intens)/(max_intens-min_intens)*255).astype(np.uint8)
            plt.imshow(img_show,cmap='gray')
        elif show_mode == "log":
            min_intens=img.min()
            max_intens=img.max()
            img_show=(np.log(img-min_intens+1e-5)/np.log(max_intens-min_intens)*255).astype(np.uint8)
            plt.imshow(img_show,cmap='gray')
    if not show_axis:
        plt.axis('off')
    if return_fig: return fig 


def scale(image:np.ndarray, ratio:float)->np.ndarray:
    """アフィン変換を使って拡大縮小
    アフィン変換について:https://qiita.com/koshian2/items/c133e2e10c261b8646bf

    Args:
        image (np.ndarray): 拡大縮小したい画像
        ratio (float): 拡大縮小の割合。変換後の高さ/元の高さ

    Returns:
        np.ndarray: 拡大縮小した画像
    """
    assert ratio > 0
    h, w = image.shape[:2]
    h_scaled=int(h*ratio)
    w_scaled=int(w*ratio)
    # cv2.getAffineTransformは変換元と変換後の3点を指定すれば、
    # 2X3のアフィン変換行列を生成する
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w_scaled,h_scaled), cv2.INTER_LANCZOS4) # 補間法も指定できる

def histeq(image:np.ndarray,nbr_bins:int=256)->tuple[np.ndarray,np.ndarray]:
    """グレースケールの画像のヒストグラムを平坦化する。

    Args:
        image (np.ndarray): 平坦化したいグレースケール画像
        nbr_bins (int, optional): 補間の細かさ. Defaults to 256.

    Returns:
        tuple[np.ndarray,np.ndarray]: 平坦化した画像と、元画像の累積分布関数(CDF)
    """
    assert len(image.shape) == 2
    # 画像のヒストグラムを得る
    imhist,bins=np.histogram(image.flatten(),bins=nbr_bins,density=True)
    cdf=imhist.cumsum() # 累積和
    cdf=255*cdf/cdf[-1] # 正規化

    # cdfを線形補間し、新しいピクセル値とする。
    im2=np.interp(image.flatten(),bins[:-1],cdf)
    # np.interp(x,xp,fp)でxの点を(xp、fp)という組み合わせの関数で補間する
    # (つまりy=f(x)という関数があったとして、xpの点の値fpしかわかっていない状況)

    # 今回imageで出現する明るさの点を、cdfに基づいて補間する
    # そうすると出現回数が多い明るさはばらける

    # ある明るさの付近が出現回数が多い
    # →その付近は元のヒストグラムで高い山となる
    # →そのヒストグラムを累積和したcdfの傾きが急になる
    # →補間されるときに明るさがばらけやすい
    return im2.reshape(image.shape),cdf

def imread(
        filename:str,
        flags:int=cv2.IMREAD_COLOR,
        dtype:type=np.uint8):
    """ 
    Windows上で動かす場合、cv2.imreadは日本語ファイル名に対応していない(アスキー文字のみ対応)ので、
    Windows上で日本語ファイル名のファイルを読み込む際の関数
    
    Args:
        filename (str): ファイル名orファイルパス
        flags (int, optional): imdecodeの引数. Defaults to cv2.IMREAD_COLOR.
        dtype (type, optional): fromfileの引数。dtype. Defaults to np.uint8.

    Returns:
        np.ndarray: 読み込んだarrayのimg
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    

def imwrite(filename:str, img:np.ndarray, params:tuple=None):
    """
    Windows上で動かす場合、cv2.imwriteは日本語ファイル名に対応していない(アスキー文字のみ対応)ので、
    Windows上で日本語ファイル名でファイルを書き込む際の関数

    Args:
        filename (str): ファイル名 or ファイルパス
        img (np.ndarray): 保存したい画像
        params (tuple, optional): cv2.imencodeのパラメータ. Defaults to None.

    Returns:
        bool: 書き込みが成功したかどうか
    """
    try:
        ext = Path(filename).suffix.lstrip(".")
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False