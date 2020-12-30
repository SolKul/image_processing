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
        show_as_it_is=False,
        show_axis=False):
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
        if show_as_it_is:
            img_show=img.astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray',vmax=255)
        else:
            min_intens=img.min()
            max_intens=img.max()
            img_show=((img-min_intens)/(max_intens-min_intens)*255).astype(np.uint8)
            plt.figure(figsize=figsize)
            plt.imshow(img_show,cmap='gray')
    if not(show_axis):
        plt.axis('off')
    plt.show()


def scale(image, ratio):
    """
    アフィン変換を使って拡大縮小
    """
    h, w = image.shape[:2]
    # アフィン変換のもととなる行列
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (2*w, 2*h), cv2.INTER_LANCZOS4) # 補間法も指定できる