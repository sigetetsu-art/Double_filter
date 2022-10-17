import numpy as np
import padding as pad

def lsm(A, b):
    return (np.linalg.inv(A.T * A) * A.T * b)

def calc_filter_init(original_image, degraded_image, FILTER_LENGTH, PADDING_SIZE):
    FILTER_SIZE = FILTER_LENGTH * FILTER_LENGTH
    h, w = degraded_image.shape
    filter_target = []
    original_pixels = np.matrix(original_image).reshape(-1, 1) #列が1列の配列にする
    for y in range(PADDING_SIZE, h - PADDING_SIZE):
        for x in range(PADDING_SIZE, w - PADDING_SIZE):
            filter_target.append(np.array((degraded_image[y - PADDING_SIZE: y + PADDING_SIZE + 1, x - PADDING_SIZE: x + PADDING_SIZE + 1]).reshape(FILTER_SIZE), dtype = "float64"))
    filter_target = np.matrix(filter_target)
    filter = (lsm(filter_target, original_pixels)).reshape(FILTER_LENGTH, FILTER_LENGTH)
    return filter

def calc_filter_init2(original_image, degraded_image, FILTER_LENGTH, PADDING_SIZE, Canny_IMAGE):
    FILTER_SIZE = FILTER_LENGTH * FILTER_LENGTH
    h, w = degraded_image.shape
    filter_target = []
    filter_target2 = []
    original_target1 = []
    original_target2 = []
    original_image = pad.mirror_padding(original_image, PADDING_SIZE)
    for y in range(PADDING_SIZE, h - PADDING_SIZE):
        for x in range(PADDING_SIZE, w - PADDING_SIZE):
            if(Canny_IMAGE[y][x] == 255): #エッジ部分のフィルタ
                filter_target.append(np.array((degraded_image[y - PADDING_SIZE: y + PADDING_SIZE + 1, x - PADDING_SIZE: x + PADDING_SIZE + 1]).reshape(FILTER_SIZE), dtype = "float64"))
                original_target1.append(original_image[y][x])
            elif(Canny_IMAGE[y][x] == 0):
                filter_target2.append(np.array((degraded_image[y - PADDING_SIZE: y + PADDING_SIZE + 1, x - PADDING_SIZE: x + PADDING_SIZE + 1]).reshape(FILTER_SIZE), dtype = "float64"))
                original_target2.append(original_image[y][x])
                
    # original_pixels = np.matrix(original_image).reshape(-1, 1) #列が1列の配列にする
    original_target1 = np.matrix(original_target1).reshape(-1, 1)
    original_target2 = np.matrix(original_target2).reshape(-1, 1)
    filter_target = np.matrix(filter_target)
    filter = (lsm(filter_target, original_target1)).reshape(FILTER_LENGTH, FILTER_LENGTH)
    filter_target2 = np.matrix(filter_target2)
    filter2 = (lsm(filter_target2, original_target2)).reshape(FILTER_LENGTH, FILTER_LENGTH)
    return filter, filter2

def lsm_point_symmetry2(original_image, degraded_img, FILTER_LENGTH, PADDING_SIZE, Canny_Image):
    FILTER_SIZE = FILTER_LENGTH*FILTER_LENGTH
    h, w = degraded_img.shape
    filter_target = []
    filter_target2 = []
    original_target1 = []
    original_target2 = []
    original_image = pad.mirror_padding(original_image, PADDING_SIZE)
    
    for y in range(PADDING_SIZE, h-PADDING_SIZE):
        for x in range(PADDING_SIZE, w-PADDING_SIZE):
            if(Canny_Image[y][x] == 255):
                tmp1 = np.array((degraded_img[y-PADDING_SIZE:y+PADDING_SIZE+1, x-PADDING_SIZE:x+PADDING_SIZE+1]).reshape(FILTER_SIZE), dtype='float64')
                tmp2 = tmp1[:int((FILTER_SIZE-1)/2)] + tmp1[:int((FILTER_SIZE-1)/2):-1]
                tmp2 = np.append(tmp2, tmp1[int((FILTER_SIZE-1)/2)])
                filter_target.append(tmp2)
                original_target1.append(original_image[y][x])
            elif(Canny_Image[y][x] == 0):
                tmp3 = np.array((degraded_img[y-PADDING_SIZE:y+PADDING_SIZE+1, x-PADDING_SIZE:x+PADDING_SIZE+1]).reshape(FILTER_SIZE), dtype='float64')
                tmp4 = tmp3[:int((FILTER_SIZE-1)/2)] + tmp3[:int((FILTER_SIZE-1)/2):-1]
                tmp4 = np.append(tmp4, tmp3[int((FILTER_SIZE-1)/2)])
                filter_target2.append(tmp4)
                original_target2.append(original_image[y][x])
                
    filter_target = np.matrix(filter_target)
    filter_target2 = np.matrix(filter_target2)
    original_target1 = np.matrix(original_target1).reshape(-1, 1)
    original_target2 = np.matrix(original_target2).reshape(-1, 1)
    
    filter = np.array(lsm(filter_target, original_target1))
    filter2 = np.array(lsm(filter_target2, original_target2))

    filterSymmetry1 = filter
    filterSymmetry2 = filter2
    for i in range(len(filter)-1):
        filterSymmetry1 = np.append(filterSymmetry1, filter[-2 - i])
        filterSymmetry2 = np.append(filterSymmetry2, filter2[-2 - i])
        
    filterSymmetry1 = filterSymmetry1.reshape(FILTER_LENGTH, FILTER_LENGTH)
    filterSymmetry2 = filterSymmetry2.reshape(FILTER_LENGTH, FILTER_LENGTH)

    return filterSymmetry1, filterSymmetry2
