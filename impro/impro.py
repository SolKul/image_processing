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
        return_fig:bool=False):
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

    Returns:
        plt.Fiugre(optional):ax=fig.axes[0]とすれば、その後表示を付け足せる。
    """
    fig=plt.figure(figsize=figsize)
    dim_num= len(img.shape)
    # 画像の次元数は2(グレースケール)か3(カラー)のどちらかに対応
    assert 2 <= dim_num <= 3
    if dim_num==3:
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


def scale(image, ratio):
    """
    アフィン変換を使って拡大縮小
    """
    h, w = image.shape[:2]
    # cv2.getAffineTransformは変換元と変換後の3点を指定すれば、
    # 2X3のアフィン変換行列を生成する
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (2*w, 2*h), cv2.INTER_LANCZOS4) # 補間法も指定できる

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