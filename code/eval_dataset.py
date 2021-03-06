import os
from PIL import Image
import numpy as np


dataset_path = '../../../datasets_local/Datasets/segmentation/any_dataset'
dataset_name = 'Name to print'
subsets = ['train', 'valid', 'test']
max_class = 30
count_res = {}

for sst in subsets:
    by_classes = np.zeros(max_class)
    full_path = dataset_path + '/' + sst + '/masks'
    if os.path.exists(full_path):
        for img_fname in os.listdir(full_path):
            full_file = full_path + '/' + img_fname
            img = Image.open(full_file)
            im_mat = np.asarray(img)
            by_classes += np.array([np.count_nonzero(im_mat == i)
                                    for i in range(max_class)])
        classes_per_img = by_classes / len(os.listdir(full_path))
        count_res[sst] = {'Pixels by class' : by_classes,
                          'Pixels by class per image' : classes_per_img,
                          'Number of images' : len(os.listdir(full_path))}

print(dataset_name + ' results:')
print(count_res)
