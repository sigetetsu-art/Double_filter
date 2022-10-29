import sys
import cv2
import numpy as np
from scipy import signal
from PIL import Image
import lsm
import psnr

#Constant
#ユーザー定義定数
L = 255
K1 = 0.01
K2 = 0.03
C1 = (K1 * L) ** 2
C2 = (K2 * L) ** 2
SSIM_LENGTH = 8

#よく変えるパラメータ
FILTER_LENGTH = 3 #フィルタサイズ
SYMMETRY = 2 #0 : Normal 1:Point symmetry 2:Mirror symmetry
BIT = 12 #量子化ビット数

#フィルタ，パディングサイズ
FILTER_SIZE = FILTER_LENGTH * FILTER_LENGTH
PADDING_SIZE = int((FILTER_LENGTH - 1) / 2)#パディング処理のサイズを決める

#方向ベクトル DirectionVector：DIV
DIV = [[y, x]
        for y in range(-PADDING_SIZE, PADDING_SIZE + 1)
        for x in range(-PADDING_SIZE, PADDING_SIZE + 1)]

#パッチ毎の画素値平均，分散所得用のフィルタ
WINDOW1 = np.ones((SSIM_LENGTH, SSIM_LENGTH)) / SSIM_LENGTH ** 2 #パッチ内の平均を求める用
WINDOW2 = np.ones((SSIM_LENGTH, SSIM_LENGTH)) / (SSIM_LENGTH ** 2 - 1) #パッチ内の分散，共分散を求める用

def mirror_padding(image, PADDING_SIZE):
    image_pixels = np.array(image)
    padding_image = np.pad(image_pixels, ((PADDING_SIZE, PADDING_SIZE), (PADDING_SIZE, PADDING_SIZE)), "edge")
    return padding_image

def filtering(padding_image, filter):
    filtered_img = signal.fftconvolve(filter[::-1,::-1], padding_image, mode='valid')
    filtered_img[filtered_img < 0] = 0
    filtered_img[filtered_img > 255] = 255
    filtered_img = (filtered_img * 2 + 1) // 2
    filtered_img = np.array(filtered_img, dtype="float64")
    return filtered_img


def filtering2(degraded_image, filter1, filter2, Canny_image):
    filter1 = np.array(filter1)
    filter2 = np.array(filter2)
    h, w = degraded_image.shape #h:画像の高さ w:画像の幅 
    filtered_image = np.zeros((h - 2 * PADDING_SIZE, w - 2 * PADDING_SIZE))
    
    for y in range(PADDING_SIZE, h - PADDING_SIZE):
        for x in range(PADDING_SIZE, w - PADDING_SIZE):
            if(Canny_image[y][x] == 255): #エッジ用のフィルタ処理
                filtered_image[y - PADDING_SIZE][x - PADDING_SIZE] = np.sum(degraded_image[y - PADDING_SIZE : y + PADDING_SIZE + 1, x - PADDING_SIZE : x + PADDING_SIZE + 1] * filter1)
            elif(Canny_image[y][x] == 0):
                filtered_image[y - PADDING_SIZE][x - PADDING_SIZE] = np.sum(degraded_image[y - PADDING_SIZE : y + PADDING_SIZE + 1, x - PADDING_SIZE : x + PADDING_SIZE + 1] * filter2)

    filtered_image[filtered_image < 0] = 0
    filtered_image[filtered_image > 255] = 255
    filtered_image = (filtered_image * 2 + 1) // 2
    filtered_image = np.array(filtered_image, dtype = "float64")
    print(filtered_image)
    return filtered_image
            

def constant(original_img, degraded_img):
    #パディング済みの劣化画像のパッチ毎の画素平均 pad_mu
    pad_mu = signal.fftconvolve(WINDOW1, degraded_img, mode='valid') #パッチ内の画素の平均を求める
    patch_h, patch_w = pad_mu.shape
    h, w = degraded_img.shape
    #原画像平均 original_mu
    original_mu = signal.fftconvolve(WINDOW1, original_img, mode='valid')
    #原画像分散　original_sigma_sq
    original_sigma_sq = signal.fftconvolve(WINDOW2, original_img*original_img, mode='valid') - original_mu**2*(SSIM_LENGTH**2)/(SSIM_LENGTH**2-1)
    #各フィルタ係数方向にスライドさせた画素値 slide_pels
    slide_pels_list = np.empty((0, h-PADDING_SIZE*2, w-PADDING_SIZE*2))
    alfa_list = np.empty((0, patch_h-PADDING_SIZE*2, patch_w-PADDING_SIZE*2))#松田先生の資料参照
    gamma_list = np.empty((0, patch_h-PADDING_SIZE*2, patch_w-PADDING_SIZE*2))#松田先生の資料参照
    for i in range(FILTER_SIZE):
        #slide_pels
        slide_pels = degraded_img[PADDING_SIZE+DIV[i][0]:h-PADDING_SIZE+DIV[i][0], PADDING_SIZE+DIV[i][1]:w-PADDING_SIZE+DIV[i][1]]
        slide_pels_list = np.vstack((slide_pels_list, [slide_pels])) #vstack・・・arrayを縦方向に連結する
        #alfa
        alfa = pad_mu[PADDING_SIZE+DIV[i][0]:patch_h-PADDING_SIZE+DIV[i][0], PADDING_SIZE+DIV[i][1]:patch_w-PADDING_SIZE+DIV[i][1]]
        alfa_list = np.vstack((alfa_list, [alfa]))
        #gamma
        #gamma = signal.fftconvolve(WINDOW2, original_img*slide_pels, mode='valid') - original_mu*alfa_list[i]*(SSIM_LENGTH**2)/(SSIM_LENGTH**2-1)
        #gamma_list = np.vstack((gamma_list, [gamma]))

    return (original_mu, original_sigma_sq, alfa_list, slide_pels_list)

def ssim_cal(original_img, filtered_img, original_mu=None, original_sigma_sq=None):
    if type(original_mu) or type(original_sigma_sq) == type(None):
        #原画像平均 original_mu
        original_mu = signal.fftconvolve(WINDOW1, original_img, mode='valid')
        #原画像分散　original_sigma_sq
        original_sigma_sq = signal.fftconvolve(WINDOW2, original_img*original_img, mode='valid') - original_mu**2*SSIM_LENGTH**2/(SSIM_LENGTH**2-1)
    #再生画像平均 filtered_mu
    filtered_mu = signal.fftconvolve(WINDOW1, filtered_img, mode='valid')
    #再生画像分散　filtered_sigma_sq
    filtered_sigma_sq = signal.fftconvolve(WINDOW2, filtered_img*filtered_img, mode='valid') - filtered_mu**2*(SSIM_LENGTH**2)/(SSIM_LENGTH**2-1)
    #共分散　covariance
    covariance = signal.fftconvolve(WINDOW2, original_img*filtered_img, mode='valid') - original_mu*filtered_mu*(SSIM_LENGTH**2)/(SSIM_LENGTH**2-1)
    #SSIM
    ssim = (2*original_mu*filtered_mu+C1)*(2*covariance+C2)/((original_mu**2+filtered_mu**2+C1)*(original_sigma_sq+filtered_sigma_sq+C2))
    #print("SSIM：%s" %np.mean(ssim))

    return (ssim, filtered_mu, filtered_sigma_sq, covariance)

def img2canny(image):
    return np.array(cv2.Canny(image, 10, 100))

def Border_detection(degraded_image):
    h, w = degraded_image.shape
    binary_image = np.zeros(degraded_image.shape)
    
    for i in range(h):
        for j in range(w):
            if(i % 7 == 0 or j % 7 == 0):
                binary_image[i][j] = 255
    
    return binary_image

def main():
    param = sys.argv
    
    original_image = Image.open(param[1])
    original_image = np.array(original_image, dtype = "float64")
    degraded_image = Image.open(param[2])
    degraded_image = np.array(degraded_image, dtype = "float64")
    
    original_mu, original_sigma_sq, alfa_list, slide_pels_list = constant(original_image, degraded_image)
    first_ssim, _, _, _ = ssim_cal(original_image, degraded_image, original_mu, original_sigma_sq)
    print("SSIM of original image and degraded image : ", np.mean(first_ssim))
    first_PSNR = psnr.calc_psnr(original_image, degraded_image)
    print("PSNR of original image and degraded image : ", first_PSNR)

    binary_image = Border_detection(degraded_image)
    binary_image = mirror_padding(np.array(binary_image), PADDING_SIZE)
    
    degraded_image = mirror_padding(degraded_image, PADDING_SIZE)
    
    filter1, filter2 = lsm.calc_filter_init2(original_image, degraded_image, FILTER_LENGTH, PADDING_SIZE, binary_image)
    
    print("filter ------------------------------------")
    print(filter1)
    print("-------------------------------------------")
    print("filter2------------------------------------")
    print(filter2)
    print("-------------------------------------------")
    filtered_image = filtering2(degraded_image, filter1, filter2, binary_image)
    ssim, filtered_mu, filtered_sigma_sq, covariance = ssim_cal(original_image, filtered_image, original_mu, original_sigma_sq)
    print("first SSIM : ", np.mean(ssim))
    PSNR = psnr.calc_psnr(original_image, filtered_image)
    print("first PSNR : ", PSNR)
    print(filtered_image)
    filtered_image = Image.fromarray(filtered_image.astype("uint8"))
    filtered_image.show()

if __name__ == "__main__":
    main()
    