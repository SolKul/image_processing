from scipy.ndimage import filters
import numpy as np
import cv2
import matplotlib.pyplot as plt

from . import image_processing as ip

def compute_harris_response(im,sigma=3):
    """
    グレースケール画像の各ピクセルについて、
    Harris coner detectorの応答関数を計算する
    """
    
    #微分フィルタ
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy=np.zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)
    
    #Harris行列の成分を計算する
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)

    #判別式と対角成分
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    # 返り値の画像は引数の画像と同じサイズとなる。
    # Errata for Programming Computer Vision with Pythonより修正
    return Wdet/(Wtr*Wtr)

def search_harris_point(harrisim,min_dist=10,threshold=0.1):
    """
    Harris応答画像からハリス特徴量が大きいもの、
    つまりコーナーを返す
    min_distはすでにリストに追加されたコーナーや画像境界から分離する最小ピクセル数

    Return: ハリス特徴量が大きい順の、y軸(列)、x軸(行)の順で入った座標のタプルのリスト(filtered_coords)
    """
    
    #閾値thresholdを超えるコーナー候補を見つける
    corner_threshold=harrisim.max()*threshold
    harrisim_t=(harrisim > corner_threshold) *1

    # 候補の座標を得る
    # numpyのarrayは最初が列方向、行方向だが、
    # ndarray.nonzero()は行方向のインデックス、
    # 列方向のインデックスの順にタプルで返すので、
    # nonzero()を転置している。
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
    # coordsは列方向、行方向のタプルのリスト
    filtered_coords = []
    for i in index:
        # allowed_locationsで1が入っている点だけを調べる
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            # ある点がリストに追加されたら場合は
            # そこからmin_dist四方はリストに追加されないようにする。
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                             (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    """
    画像中に見つかったコーナーを描画 
    """
    im_draw=image.copy()
    for coord in filtered_coords:
        # coordはy軸(列)、x軸(行)の順だが、
        # cv2.circleは座標はx軸(行)、y軸(列)の順
        pt=(coord[1],coord[0])
        cv2.circle(im_draw,pt,2,(0,100,200),thickness=20)
    ip.show_img(im_draw)


def pick_pixels_near_coord(image,coords,wid=5):
    """
    各点について、点の周辺で幅2*wid+1の近傍ピクセル値を取り出す
    (search_harris_pointでの点の最小距離 min_dist > widを仮定している)
    
    return:
        特徴点周辺の画素のarrayのリスト(desc)
    """
    desc=[]
    for coord in coords:
        patch=image[(coord[0]-wid):(coord[0]+wid+1),
                    (coord[1]-wid):(coord[1]+wid+1)].flatten()
        desc.append(patch)
    return desc
 
def pick_max_ncc_indices(desc1,desc2,threshold=0.5):
    """
    正規化相互相関(nomal cross correlation)を用いて、第1の画像の各コーナー点記述子と、
    第2の画像の記述子でnccが最大の点のindexを返す

    Args:
        desc1 (list):画像1の特徴点周辺の画素のarrayのリスト
        desc2 (list):画像2の特徴点周辺の画素のarrayのリスト
    """
    n=len(desc1[0])

    # 対応点ごとの距離
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = np.sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    ndx = np.argsort(-d)
    matchs_indices = ndx[:,0]
    #matchs_indicesは画像1と画像2の記述子のうち、
    #画像1から画像2への相関関数を大きい順に並べたindex
    return matchs_indices

def pick_same_matches(desc1,desc2,threshold=0.5):
    """
    extract_max_ncc_index()で取り出したindexが双方向で一致しているかを調べ、
    一致していたものだけのindexを返す。
    """
    matches_12=pick_max_ncc_indices(desc1,desc2,threshold)
    matches_21=pick_max_ncc_indices(desc2,desc1,threshold)
    #matchs_12は画像1と画像2の記述子のうち、
    #画像1から画像2への相関関数が最大のもののindex

    #非負のindex=相関関数がthreshold以上で最大だったindexのみ取り出す
    ndx_12= np.where(matches_12 >= 0)[0]

    #画像1のあるindex、nに対し、相関関数が最大だったmatches_12[n]は
    #双方向で相関関数が最大であれば、
    #画像2→1のmatches_21[matches_12[n]]はnになるはず
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

def plot_matches(
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
        locs1 (list): y軸(列)、x軸(行)の順で入ったコーナーの座標のタプルのリスト
        locs2 (list): y軸(列)、x軸(行)の順で入ったコーナーの座標のタプルのリスト
    """
    
    im3 = concatenate_img_horiz(im1,im2)
    # show_belowがTrueなら、線で結んでない画像を下に表示する。
    if show_below:
        im3 = np.vstack((im3,im3))
    
    ip.show_img(
        im3,
        show_mode='as_it_is',
        figsize=figsize,
        successive_plot=True)
    width1 = im1.shape[1]
    for i,m in enumerate(match_indices):
        # 対応点が見つからなければm=-1で、
        # 対応点が見つかった点についてプロット。
        if m>=0:
            # x軸、y軸の順でプロット、
            # 横に並べているので+width1とする。
            plt.plot(
                [locs1[i][1],locs2[m][1]+width1],
                [locs1[i][0],locs2[m][0]],
                'c')
    plt.axis('off')
    plt.show()
    
def compute_harris_ncc_and_plot(
        im1,
        im2,
        min_dist=100,
        wid=5,
        sigma=5,
        threshold=0.5,
        figsize=(10,20)):
    """
    2つの画像のハリス特徴量が大きかったコーナーを総当りで、
    マッチングし、線で結んで表示する。
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
    match_indices=pick_same_matches(d1,d2,threshold=threshold)

    # 図示する。
    plot_matches(
        im1,
        im2,
        filtered_coords1,
        filtered_coords2,
        match_indices,
        figsize=figsize)

def akaze_matching(im1,im2,threshold=0.3):
    """
    A-KAZEによる特徴量マッチング
    """
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()

    # 特徴点とその特徴量ベクトルのリスト
    kp1, des1 = akaze.detectAndCompute(im1,None)
    kp2, des2 = akaze.detectAndCompute(im2,None)

    # Brute-Force Matcher生成
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 比較機で対応オブジェクトのリストmatchesを作成
    matches = bf.match(des1, des2)
    # 対応オブジェクトのメンバのdistanceでソートする
    # sortedのkeyに無名関数(x.distanceを返す)を割り当ててソート
    matches = sorted(matches, key = lambda x:x.distance)

    # 許容最大の距離
    max_dis=matches[-1].distance*threshold
    matches_num=len(matches)
    for i in range(matches_num):
        if matches[i].distance > max_dis:
            break
    num_lim=i

    img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:num_lim], None, flags=2)
    
    return img3,kp1,kp2,matches,num_lim