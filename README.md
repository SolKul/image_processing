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