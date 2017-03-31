import h5py
from fuel.datasets.hdf5 import H5PYDataset
import os
from scipy.misc import imread

image_dim = 32
size_train = 1281149
size_val = 49999
path_train = 'dataset/train'
path_val = 'dataset/val'
BATCH = 25000
d = {
    path_train: size_train,
    path_val: size_val
}

f = h5py.File('coco_cropped.h5', mode='w')
features = f.create_dataset('features', (size_train+size_val, 3, 64, 64), dtype='float32')
targets = f.create_dataset('targets', (size_train+size_val, 3, 32, 32), dtype='float32')
split_dict = {
    'train': {
        'features': (0, size_train),
        'targets': (0, size_train)
    },
    'valid': {
        'features': (size_train, size_train+size_val),
        'targets': (size_train, size_train+size_val),
    }
}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

for direc in [path_train, path_val]:
    filenames = sorted(os.listdir(direc+'/input'))
    for i, filename in enumerate(filenames):
        image = imread(direc+'/input/'+filename).reshape((1, 3, 64, 64))
        features[i, :, :, :] = image
        image = imread(direc+'/target/'+filename).reshape((1, 3, 32, 32))
        targets[i, :, :, :] = image
        if i % BATCH == 0:
            f.flush()
            
    f.flush()
    print '{} set done'.format('Training' if direc == path_train else 'Validation')
f.flush()
f.close()
