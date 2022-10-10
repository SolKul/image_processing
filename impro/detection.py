from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt

from . import impro

def compute_harris_response(im:np.ndarray,sigma:int=3,k:float=0.04)->np.ndarray:
    """グレースケール画像の各ピクセルについて、
    Harris coner detectorの応答関数を計算する

    Args:
        im (np.ndarray): Harris応答関数を計算するグレー画像
        sigma (int, optional): ガウシアン微分フィルタのsigma. Defaults to 3.
        k (float, optional): HarrisのRの係数のカッパ. Defaults to 0.04.

    Raises:
        ValueError: 画像がグレースケールでないときのエラー

    Returns:
        np.ndarray: ハリス応答関数
    """

    # 画像はグレースケールであること
    if len(im.shape) != 2:
        raise ValueError("image should be gray scale")
    
    #微分フィルタ
    imx:np.ndarray=ndimage.gaussian_filter(im,(sigma,sigma),(0,1),float)
    imy:np.ndarray=ndimage.gaussian_filter(im,(sigma,sigma),(1,0),float)
    
    #Harris行列の成分を計算する
    Wxx = ndimage.gaussian_filter(imx*imx,sigma)
    Wxy = ndimage.gaussian_filter(imx*imy,sigma)
    Wyy = ndimage.gaussian_filter(imy*imy,sigma)

    #判別式と対角成分
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    # 返り値の画像は引数の画像と同じサイズとなる。
    # Errata for Programming Computer Vision with Pythonより修正
    # https://www.oreilly.com/catalog/errata.csp?isbn=0636920022923
    # r_harris = Wdet/(Wtr*Wtr)

    #　↑だとWtrが0の場合にnanになってしまうので、元のHarrisの定義通りにする
    r_harris = Wdet-k*Wtr**2
    return r_harris

def search_harris_point(harrisim:np.ndarray,min_dist:int=10,threshold:float=0.1)->list[np.ndarray]:
    """Harris応答画像からハリス特徴量が大きいもの、
    つまりコーナーを返す

    Args:
        harrisim (np.ndarray): Harris応答を計算した画像
        min_dist (int, optional): すでにリストに追加されたコーナーや画像境界から分離する最小ピクセル数. Defaults to 10.
        threshold (float, optional): コーナーとして認定する割合閾値. Defaults to 0.1.

    Returns:
        list[np.ndarray]: ハリス特徴量が大きい順の、y座標(縦方向)、x座用(横方向)の順で入った座標のタプルのリスト
    """
    
    #閾値thresholdを超えるコーナー候補を見つける
    corner_threshold=harrisim.max()*threshold
    harrisim_t=(harrisim > corner_threshold) *1

    # 候補の座標を得る
    # ndarray.nonzero()はy座標の1次元配列、x座標の1次元配列の順に入ったタプルを返すので、
    # arrayにして転置することで、[y,x]という座標が縦方向に並んだ2次元配列になる
    coords = np.array(harrisim_t.nonzero()).T

    #候補の値を得る
    candidate_values = [harrisim[c[0],c[1]] for c in coords]

    #候補をソートする
    # Errata for Programming Computer Vision with Pythonより修正
    index = np.argsort(candidate_values)[::-1]

    #画像中で、そこのコーナーを選んでいいと
    #許容する点の座標を配列に格納する
    #画像境界からmin_dist分は許容しない
    allowed_locations=np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1

    # 最小距離を考慮しながら、最良の点を得る
    # coordsは縦方向、横方向のタプルのリスト
    filtered_coords:list[np.ndarray] = []
    for i in index:
        # allowed_locationsで1が入っている点だけを調べる
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            # ある点がリストに追加されたら場合は
            # そこからmin_dist四方はリストに追加されないようにする。
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                             (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords

def plot_harris_points(image:np.ndarray,filtered_coords:list[np.ndarray]):
    """画像中に見つかったコーナーを描画 

    Args:
        image (np.ndarray): 元の画像
        filtered_coords (list[np.ndarray]): 検出されたコーナーの座標のリスト.[y,x]という順番で格納されている。
    """
    im_draw=image.copy()
    for coord in filtered_coords:
        # coordはy軸(列)、x軸(行)の順だが、
        # cv2.circleは座標はx軸(行)、y軸(列)の順
        pt=(coord[1],coord[0])
        cv2.circle(im_draw,pt,2,(0,100,200),thickness=20)
    impro.imshow(im_draw)


def pick_pixels_near_coord(image:np.ndarray,coords:list[np.ndarray],wid:int=5)->list[np.ndarray]:
    """各点について、点の周辺で幅2*wid+1の近傍ピクセル値を取り出す
    (search_harris_pointでの点の最小距離 min_dist > widを仮定している)


    Args:
        image (np.ndarray): _description_
        coords (list[np.ndarray]): _description_
        wid (int, optional): _description_. Defaults to 5.

    Returns:
        list[np.ndarray]: 特徴点周辺の画素のarrayのリスト(desc)
    """
    desc=[]
    for coord in coords:
        patch=image[(coord[0]-wid):(coord[0]+wid+1),
                    (coord[1]-wid):(coord[1]+wid+1)].flatten()
        desc.append(patch)
    return desc
 
def calc_max_ncc_indices(desc1:list[np.ndarray],desc2:list[np.ndarray],threshold:float=0.5):
    """
    正規化相互相関(nomal cross correlation)を用いて、第1の画像の各コーナー点記述子と、
    第2の画像の記述子でnccが最大の点のindexを返す

    Args:
        desc1 (list):画像1の特徴点周辺の画素のarrayのリスト
        desc2 (list):画像2の特徴点周辺の画素のarrayのリスト
    """
    n=len(desc1[0])

    # 対応点ごとの距離(類似してるほど小さい)
    # 最初は全ての距離を1とする
    d = np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = np.sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                # nccが大きい→距離は小さいのでマイナスにして格納
                d[i,j] = - ncc_value
    # dの行方向、つまりあるdesc1について全てのdesc2について、
    # 距離が小さい(類似度が高い)座標を見つけ出す
    ndx = np.argsort(d)
    matchs_indices = ndx[:,0]
    # matchs_indicesは画像1と画像2の記述子のうち、
    # 画像1から画像2への相関関数を大きい順に並べたindex
    # matchs_indicesの
    # 1つ目のindexはdesc1[0]と相関係数が一番大きいdesc2のindex、
    # 2つ目のindexはdesc1[1]と相関係数が一番大きいdesc2のindex、
    # ...というようになる。
    return matchs_indices

def calc_same_matches(desc1,desc2,threshold=0.5):
    """
    calc_max_ncc_indices()で取り出したindexが双方向で一致しているかを調べ、
    一致していたものだけのindexを返す。
    """
    matches_12=calc_max_ncc_indices(desc1,desc2,threshold)
    matches_21=calc_max_ncc_indices(desc2,desc1,threshold)
    # matchs_12は画像1と画像2の記述子のうち、
    # 画像1から画像2への相関関数が最大のもののindex

    # indexがマイナスだったら双方向の一致は飛ばす
    ndx_12= np.where(matches_12 >= 0)[0]

    # 画像1のdesc1のn番目に対し、
    # 相関関数が最大の画像2のdesc2はmatches_12[n]番目になる。
    # 双方向で相関関数が最大であれば、
    # 画像2→1のmatches_21[matches_12[n]]はnになるはず
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
            
    return matches_12

def concatenate_img_horiz(im1,im2):
    """
    2つの画像を横に並べた画像を返す
    """
    height1 = im1.shape[0]
    height2 = im2.shape[0]
    
    im1_cvt=im1.copy()
    im2_cvt=im2.copy()

    dimension1=len(im1.shape)
    dimension2=len(im2.shape)

    #もし画像同士で次元が違ったら、合わせる。
    if (dimension1==3) & (dimension2==2):
        im2_cvt=cv2.cvtColor(im2_cvt,cv2.COLOR_GRAY2BGR)
    elif (dimension1==2) & (dimension2==3):
        im1_cvt=cv2.cvtColor(im1_cvt,cv2.COLOR_GRAY2BGR)
        
    #高さが違ったら、その分0で埋める
    if height1 < height2:
        im1_cvt = np.concatenate((im1_cvt,np.zeros((height2-height1,*im1.shape[1:]))),axis=0)
    elif height1 > height2:
        im2_cvt = np.concatenate((im2_cvt,np.zeros((height1-height2,*im2.shape[1:]))),axis=0)

    return np.concatenate((im1_cvt,im2_cvt),axis=1).astype(np.uint8)

def draw_matches(
        im1,
        im2,
        locs1,
        locs2,
        match_indices,
        show_below=True,
        figsize=(6,4)):
    """
    対応点を線で結んで画像を表示する。

    Args:
        im1 (np.ndaarray): 1つ目の画像、
        im2 (np.ndaarray): 2つ目の画像
        locs1 (list): y軸(縦)、x軸(横)の順で入ったコーナーの座標のタプルのリスト
        locs2 (list): y軸(縦)、x軸(横)の順で入ったコーナーの座標のタプルのリスト
    """
    
    im3 = concatenate_img_horiz(im1,im2)
    # show_belowがTrueなら、線で結んでない画像を下に表示する。
    if show_below:
        im3 = np.vstack((im3,im3))
    
    # fig=impro.imshow(
    #     im3,
    #     show_mode='as_it_is',
    #     figsize=figsize,
    #     return_fig=True)
    # ax=fig.axes[0]
    width1 = im1.shape[1]
    # 線の太さを画像の高さから相対的に決定する
    relative_thick=int(np.ceil(im3.shape[0]/1000)*3)
    for i,m in enumerate(match_indices):
        # 対応点が見つからなければm=-1で、
        # 対応点が見つかった点についてプロット。
        if m>=0:
            # x軸、y軸の順でプロット、
            # 横に並べているので+width1とする。
            cv2.line(
                im3,
                (locs1[i][1],locs1[i][0]),
                (locs2[m][1]+width1,locs2[m][0]),
                color=(0,0,255),
                thickness=relative_thick,
                lineType=cv2.LINE_AA)
    return im3
    
def draw_harris_ncc_match(
        im1,
        im2,
        min_dist=100,
        wid=5,
        sigma=5,
        threshold=0.5,
        figsize=(10,20)):
    """
    2つの画像のハリス特徴量が大きかったコーナーを総当りで、
    マッチングし、線で結んが画像を描写する。
    ただ、精度は低い。あくまで特徴量マッチングの練習
    """
    dimension1=len(im1.shape)
    dimension2=len(im2.shape)
    
    im1_cvt=im1.copy()
    im2_cvt=im2.copy()
    #もし画像がグレースケールでなかったらグレースケールに変換
    if (dimension1==3):
        im1_cvt=cv2.cvtColor(im1_cvt,cv2.COLOR_BGR2GRAY)
    if (dimension2==3):
        im2_cvt=cv2.cvtColor(im2_cvt,cv2.COLOR_BGR2GRAY)
    
    #Harris corner ditectorを計算
    harrisim = compute_harris_response(im1_cvt,sigma=sigma)
    #Harrisの応答関数が大きいものの座標を探す
    filtered_coords1 = search_harris_point(harrisim,min_dist=min_dist)
    #Cornerの座標周辺のピクセル値を取得する    
    d1 = pick_pixels_near_coord(im1_cvt,filtered_coords1,wid=wid)
    
    harrisim = compute_harris_response(im2_cvt,sigma=sigma)
    filtered_coords2 = search_harris_point(harrisim,min_dist=min_dist)
    d2 = pick_pixels_near_coord(im2_cvt,filtered_coords2,wid=wid)

    # Cornerの座標周辺のピクセル値同士を比較し、その相関関数が大きいもののindexを取得する
    match_indices=calc_same_matches(d1,d2,threshold=threshold)

    # 図示する。
    return draw_matches(
        im1,
        im2,
        filtered_coords1,
        filtered_coords2,
        match_indices,
        figsize=figsize)

def akaze_matching(
    img1:np.ndarray,
    img2:np.ndarray,
)->tuple[
    tuple[cv2.KeyPoint,...],
    tuple[cv2.KeyPoint,...],
    list[cv2.DMatch],
    ]:
    """
    A-KAZEによる特徴量マッチング
    """
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()

    # 特徴点とその特徴量ベクトルのリスト
    kp1, des1 = akaze.detectAndCompute(img1,None)
    kp2, des2 = akaze.detectAndCompute(img2,None)

    # Brute-Force Matcher生成
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    # 2番めに近かったkey pointと差があるものをいいkey pointとする。
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return kp1,kp2,good

def sift_matching(img1:np.ndarray,img2:np.ndarray,threshold:float=0.3,draw_result:bool=False):
    """
    siftによる特徴量マッチング
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher_create(cv2.NORM_L2)
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    if draw_result:
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
        return kp1,kp2,good,img3
    else:
        return kp1,kp2,good