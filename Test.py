import cv2
from tkinter import filedialog
import numpy as np
from matplotlib import pyplot as plt

#参考
#https://camp.trainocate.co.jp/magazine/python-opencv/

def cv_main():
    cascade_path = './opencv-4.8.0/data/haarcascades/'

    path = get_file_path()
    print(path)
    img = cv2.imread(path)

    re1 = resize(img)

    triming(re1)

    image_imshow(img)

    img_gray = gray_scale(img)

    get_hist(img)

    get_threshold(img, img_gray)

    get_face_cascade(img, img_gray, cascade_path)


## fileパス取得
def get_file_path():

    typ = [('image','*.jpg'),('image','*.png')] 
    dir = 'C:\\'
    path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    return path


def resize(img):
    ########
    # リサイズ
    ########
    # 全角ファイル名はエラーになるので、注意
    re1 = cv2.resize(img, dsize=(500, 500))
    cv2.imshow('image1', re1)
    cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    cv2.destroyAllWindows()

    return re1

def triming(re1):
    ###########
    #トリミング
    ###########
    # 画像の高さ
    height = re1.shape[0]
    # 画像の幅
    width = re1.shape[1]
    # 画像サイズを取得
    print(f"height:{height}, width:{width}")
    # height:631, width:1201


    ##################
    # 画像をトリミング
    ##################
    tri_img = re1[200: 500, 300: 600]
    cv2.imshow('image2', tri_img)
    cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    cv2.destroyAllWindows()

def image_imshow(img):
    ##################
    #比率を指定して変更
    ##################
    re2 = cv2.resize(img, dsize=None, fx=1, fy=0.5)
    cv2.imshow('image3', re2)
    cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    cv2.destroyAllWindows()

def gray_scale(img):
    ##################################################
    # 画像処理のノイズ除去のため一旦グレースケールに変更
    ##################################################
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image4', img_gray)
    cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    cv2.destroyAllWindows()

    return img_gray

def get_hist(img):
    ############################
    #画像のヒストグラムを求める
    ############################
    #ヒストグラムとは、縦軸に度数、横軸に階級をとった統計グラフ
    color = ("b", "g", "r")
    #forループの中でリストやタプルなどのイテラブルオブジェクトの要素
    #と同時にインデックス番号（カウント、順番）を取得
    for i, col in enumerate(color):
        # ヒストグラムを取得
        # images：入力画像でuint8またはfloat32のデータ型で、角括弧を使って[img]で指定します
        # channels：ヒストグラムを計算する対象となるチャネルのインデックスで角括弧で指定
        # mask:マスク画像
        # histSize：ビンの数で、角括弧で指定。フルスケールは[256]
        # ranges：ヒストグラムを計算したい画像値の範囲。通常は[0,256]
        histr = cv2.calcHist(images=[img], channels=[i],
        mask=None, histSize=[256], ranges=[0, 256])
        # グラフを描画する
        plt.plot(histr, color=col)
        # x軸の範囲を設定
        plt.xlim([0, 256])
    plt.show()


def get_threshold(img, img_gray):
    ######################
    #輪郭を検出する
    ######################
    # 白黒に変換
    ret, thresh = cv2.threshold(img_gray,
    120, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("change1.png", thresh)
    # 輪郭検出
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 10)
    cv2.imwrite("change2.png", img)

def get_face_cascade(img, img_gray, cascade_path):
    ##############
    #顔画像処理
    ##############
    # 学習モデルのパス
    cascade_path_1 = cascade_path + '/haarcascade_frontalface_alt.xml'

    # 学習済みモデルの読み込み
    cascade = cv2.CascadeClassifier(cascade_path_1)

    # 顔の検出
    # minSizeで最小検出サイズを指定（今回は20*20以下は探さない）
    face_list = cascade.detectMultiScale(img_gray, minSize = (20, 20))
    # 顔が見つかるかで条件分岐
    if len(face_list):
        for (x,y,w,h) in face_list:
            # 顔が見つかった場合赤い四角で囲う
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), thickness=3)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    # 顔が見つからなかった場合
    else:
        print('見つかりませんでした')



if __name__ == "__main__":
    cv_main()