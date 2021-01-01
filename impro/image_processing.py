import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def imread(
        filename,
        flags=cv2.IMREAD_COLOR,
        dtype=np.uint8):
    """
    cv2.imreadは日本語に対応していない(アスキー文字のみ対応)ので、
    その対処
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
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
    
def show_img(
        img,
        figsize=(9,6),
        isBGR=True,
        show_mode="scale",
        show_axis=False,
        successive_plot=False):
    """
    array形式の画像を表示する。カラーでもグレースケールでも対応。
    グレースケールの場合、そのまま、スケール、ログスケールで表示が選べる。

    args:
        show_mode (str): "as_it_is":そのまま。
        "sacale" :最小値が0、最大値が255になるようスケールして
        "log" :最小値が0、最大値が255になるようログでスケールして
    """
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape)==3:
        if isBGR:
            img_cvt=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img_cvt=img
        plt.figure(figsize=figsize)
        plt.imshow(img_cvt)
    elif len(img.shape)==2:
        if show_mode == "as_it_is":
            img_show=img.astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray',vmax=255)
        elif show_mode == "scale":
            min_intens=img.min()
            max_intens=img.max()
            img_show=((img-min_intens)/np.log(max_intens-min_intens)*255).astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray')
        elif show_mode == "log":
            min_intens=img.min()
            max_intens=img.max()
            img_show=(np.log(img-min_intens+1e-5)/np.log(max_intens-min_intens)*255).astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray')
    if not show_axis:
        plt.axis('off')
    if not successive_plot:
        plt.show()


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