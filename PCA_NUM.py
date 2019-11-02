import struct
from matplotlib import pyplot as plt
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')


def read_idx1(filename):
    print(filename)
    bin_data = open(filename, 'rb').read()
    # print(bin_data)
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式
    # 标签集中，只使用2个ii。
    offset = 0
    magic_number, num = struct.unpack_from('>ii', bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num))

    # 解析数据集
    offset += struct.calcsize('>ii')
    fmt_image = '>B'
    labels = np.empty(num)
    for i in range(num):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def read_idx3(filename):
    print(filename)
    bin_data = open(filename, 'rb').read()
    # print(bin_data)
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式
    # 需要读取前4行数据，所以需要4个i。标签集中，只使用2个ii。
    offset = 0
    magic_number, num, rows, cols = struct.unpack_from('>iiii', bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num, rows, cols))
    size = rows * cols
    # 返回对应于格式化字符串'>iiii'的结构体的大小。读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    offset += struct.calcsize('>iiii')
    fmt_image = '>' + str(size) + 'B'
    # > 表示大端法
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。
    # 这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    # print(fmt_image, offset, struct.calcsize(fmt_image))

    images = np.empty((num, rows, cols))
    # 初始化图片数组
    for i in range(num):
        image_new = struct.unpack_from(fmt_image, bin_data, offset)
        images[i] = np.array(image_new).reshape((rows, cols))
        # images[i] = np.reshape(image_new, (-1, rows * cols))
        offset += struct.calcsize(fmt_image)
    return num, rows, cols, images


pic_num, images_row, images_col, train_images = read_idx3('train-images.idx3-ubyte')
# train_labels = read_idx1('train-labels.idx1-ubyte')
# t10k_images = read_idx3('t10k-images.idx3-ubyte')
# t10k_labels = read_idx1('t10k-labels.idx1-ubyte')
# print(train_images[0].shape)
'''
all_norm = np.zeros(((all_row, all_col,all_col)))
for k in range(pic_num):
    t_data = train_images[k]
    t_data_norm = np.zeros((t_data.shape[0], t_data.shape[1]))
    n = t_data.shape[1]
    mu = np.mean(t_data, axis=0)  # axis=0表示列
    sigma = np.std(t_data, axis=0)
    # print(sigma)
    for i in range(n):
        t_data_norm[:, i] = (t_data[:, i] - mu[i]) / sigma[i]

    # t_data_norm, mu, sigma = normalize(t_data)
    Sigma = np.dot(np.transpose(t_data_norm), t_data_norm) / t_data.shape[0]  # 求Sigma

    U, S, V = np.linalg.svd(Sigma)  # 求Sigma的奇异值分解
    Z = np.zeros((t_data.shape[0], 2))
    u = U[0:10000, :]  # 取前K个
    Z = np.dot(t_data_norm, u.T)  # 投影
    print(u)
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    X_rec = np.dot(Z, u)  # 还原数据（近似）
    all_norm[i] = a
    plt.imshow(X_rec, cmap='gray')
    plt.pause(0.000001)
    plt.show()
'''
for xx in [1000, 1500, 1800, 2200, 2700, 3500, 4000, 5800, 6000]:
    X = train_images[xx, :]
    n = X.shape[1]
    mu = np.zeros((1, n))
    sigma = np.zeros((1, n))
    mu = np.mean(X, axis=0)  # axis=0表示列
    sigma = np.std(X, axis=0)
    for i in range(n):
        if sigma[i] == 0:
            sigma[i] = 1
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        # print(sigma[i])
    Sigma = np.dot(np.transpose(X), X) / train_images.shape[0]
    U, S, V = np.linalg.svd(Sigma)  # 求Sigma的奇异值分解
    print(U.shape)
    Z = np.zeros((pic_num, 100))
    u = U[0:100, :]  # 取前K个
    Z = np.dot(X, u.T)  # 投影
    #  还原数据（近似）
    plt.imshow(np.dot(Z, u), cmap='gray')
    plt.pause(0.000001)
    plt.show()




