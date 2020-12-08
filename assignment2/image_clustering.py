from PIL import Image
import numpy as np
from package.kmeans import Kmeans
from package.softKmeans import SoftKmeans
from tqdm import tqdm
from glob import glob
# import matplotlib.pyplot as plt

file_list = glob('./img/resize/*_1.jpg')
i = 1
for file in tqdm(file_list):
    print("image {}".format(i))
    img = Image.open('{}'.format(file))
    img_array = np.asarray(img)/255
    for k in tqdm([2, 3, 5, 7, 10, 15, 20]):
        hard_kmeans = Kmeans(K=k, initial_mode='image', opt_method='cost_fuction')
        cluster_img = hard_kmeans.image_cluster(img_array, 'train_{}_(k={})'.format(i, k))
        hard_kmeans.distortion_measure_plot('train_{}_(k={})'.format(i, k))

    for k in tqdm([3, 10, 20, 30]):
        soft_kmeans = SoftKmeans(K=k,initial_mode='image', opt_method='cost_fuction')
        cluster_img = soft_kmeans.image_cluster(img_array, 'train_{}_(k={})'.format(i, k), type='soft_kmeans')
        soft_kmeans.distortion_measure_plot('train_{}_(k={})'.format(i, k), type='soft_kmeans')
    i += 1

# img = Image.open('./img/train1_1.jpg')
# img_array = np.asarray(img)/255
# img_array.shape
# for k in [2, 3, 5, 7, 10, 15, 20]:
#     hard_kmeans = Kmeans(K=k, initial_mode='image', opt_method='cost_fuction')
#     cluster_img = hard_kmeans.image_cluster(img_array, 'train_1_(k={})'.format(k))
#     hard_kmeans.distortion_measure_plot('train_1_(k={})'.format(k))
#
# for k in [3, 10, 15]:
#     soft_kmeans = SoftKmeans(K=k,initial_mode='image', opt_method='cost_fuction')
#     cluster_img = soft_kmeans.image_cluster(img_array, 'train_1_(k={})'.format(k), type='soft_kmeans')
#     soft_kmeans.distortion_measure_plot('train_1_(k={})'.format(k), type='soft_kmeans')
