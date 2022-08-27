# 開発環境の立ち上げ方法

## このリポジトリをクローンする

サーバ上にdockerを立ち上げる場合は、サーバ上にこのリポジトリをクローンする。ローカルPC上でdockerを立ち上げ、またはサーバのdockerにdocker contextしている場合はローカルPC上にこのリポジトリをクローンする。次のようにこのリポジトリをcloneする

```shell-session
$ git clone git@github...
```

## コンテナを立ち上げる

必要に応じて.envファイルを自分の設定に書き換えたうえで、このリポジトリの`environment`ディレクトリを開き、コンテナを立ち上げる。

コンテナを立ち上げる前に以下のコマンドで`docker-compose.yml`の設定が正しく反映できているか確認する。

```shell-session
$ docker-compose config
```

設定を確認したら次のコマンドでコンテナを立ち上げる。

```shell-session
$ docker-compose up
```

(何も設定を書き換えてなくても試験的にコンテナを立ち上げられるように、仮の共有ディレクトリ、仮のパスワードが.envファイルに書いてある。)

## Jupyter Notebookの起動

サーバ上でコンテナを立ち上げた場合は`サーバのIPアドレス:8888`、ローカルPCでコンテナを立ち上げた場合は`127.0.0.1:8888`にアクセスすることで、Jupyter Notebookにアクセスできる。パスワードはデフォルトでは`test`

## VSCodeでコンテナを開く


### サーバ上でコンテナを立ち上げた場合

VSCodeを開き拡張機能の`Remote - SSH`をインストールし、対象のサーバをVSCodeで開く。これ以降次のローカルPC上でコンテナを立ち上げた場合と同じ手順を行う。

### ローカルPC上でコンテナを立ち上げた場合

VSCodeを開き拡張機能の`Remote - Container`をインストールし、対象コンテナをVSCodeで開く。

# カメラ行列の原理

カメラ行列の原理  
まずピンホールカメラモデルについて理解  
https://www.slideshare.net/ShoheiMori/ss-64994150  

ついでに同次座標系について理解  
http://zellij.hatenablog.com/entry/20120523/p1  

要は座標を扱うときはその1次元上で扱うと便利という話  

そしてopencv公式
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html  

![カメラ行列の原理](https://docs.opencv.org/2.4/_images/pinhole_camera_model.png)

## カメラ行列について

$\boldsymbol{x}$:画像上の2次元座標  
$$
\\
\boldsymbol{x}=\left(\begin{array}{c}
x\\
y\\
1
\end{array}\right)\\
\  \\
$$  
$\boldsymbol{X}$:3次元座標  
$$
\\
\boldsymbol{X}=\left(\begin{array}{c}
X\\
Y\\
Z\\
W
\end{array}\right)\\
\  \\
$$  

ピンホールカメラを用いると、3Dの点Xから画像上の点 x（どちらも同次座標で表現）への射影は
次の式で表せる。
$$\lambda \boldsymbol{x}=P\boldsymbol{X}=K(R|t)\boldsymbol{X}$$
まずワールド座標系での3Dの点Xをカメラを中心とした座標系に
$$
(R|t)\boldsymbol{X}
$$
で射影する。  
$(R|t)$は$3\times4$なのでこの射影で同次座標系ではなく、普通の座標になるっぽい  
  
  
そして$K$で画像上の2次元座標に射影される  
  


$P=K(R|t)$  
$K$:内部パラメータ  
$$
\\
K=\left(\begin{array}{ccc}
f_x & s & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{array}\right)\\
\  \\
$$
$fx,fy$:焦点距離。厳密にはレンズの焦点距離とは違うので注意。縦横で異なることがあるのでfx,fy別になっている。  
$c_x,c_y$左端から、画像中心までの距離。$x$を左端からの座標系に戻すために必要  
$R$:回転行列(3×3)  
$t$:並行移動(3×1)  
$P$:カメラ行列、3次元座標を画像上の2次元座標に射影する行列  
$\lambda,W$:よくわからん

行列式と行列の積の関係
$$
C=AB\\
|C|=|A||B|\\
$$
カメラ行列$K$とその対角成分の符号を取ったもの$T$
$$
\\
K=\left(\begin{array}{ccc}
f_x & s & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{array}\right)\\
\  \\
\  \\
T=\left(\begin{array}{ccc}
sgn(f_x) & 0 & 0\\
0 & sgn(f_y) & 0\\
0 & 0 & sgn(1)
\end{array}\right)\\
\ \\
\ \\
sgn(|K|)=sgn(|T|)\\
\ \\
$$
その積$K'$の行列式は必ず正
$$
K'=KT\\
\ \\
if\ |T|>0\\
sgn(|K|)=sgn(|T|)>0\\
|K'|=|K||T|>0\\
\ \\
else\ |T|<0\\
sgn(|K|)=sgn(|T|)<0\\
|K'|=|K||T|>0\\
$$
