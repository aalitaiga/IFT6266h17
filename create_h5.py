import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
import os
from scipy.misc import imread
import glob

image_dim = 32
size_train = 1281149
size_val = 49999
path_train = '64_64/train2014'
path_val = '64_64/val2014'
BATCH = 5000
d = {
    path_train: size_train,
    path_val: size_val
}

f = h5py.File('coco_cropped.h5', mode='w')
image_features = f.create_dataset('features', (size_train+size_val, 3, image_dim, image_dim), dtype='float32')
split_dict = {
    'train': {
        'features': (0, size_train),
    },
    'valid': {
        'features': (size_train, size_train+size_val),
    }
}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
i = 0

for direc in [path_train, path_val]:
    l = []
    filenames = sorted(os.listdir(direc))
    for filename in filenames:
        image = imread(direc+filename).reshape((1, 3, image_dim, image_dim))
        l.append(image)
        if len(l) == BATCH:
            image_features[i:i+BATCH, :, :, :] = np.concatenate(l, axis=0)
            f.flush()
            l = []
            i += BATCH
            print i // BATCH
    image_features[i:i+len(l), :, :, :] = np.concatenate(l, axis=0)
    i += len(l)
    print '{} set done'.format('Training' if direc == path_train else 'Validation')
f.flush()
f.close()
