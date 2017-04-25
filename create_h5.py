from __future__ import print_function

import h5py
from fuel.datasets.hdf5 import H5PYDataset
import os
from scipy.misc import imread

size_train = 82611
size_val = 40438
path_train = '/Tmp/alitaiga/ift6266/processed/train/'
path_val = '/Tmp/alitaiga/ift6266/processed/valid/'
BATCH = 25000

f = h5py.File('/data/lisatmp3/alitaiga/ift6266/coco_cropped.h5', mode='w')
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
count = 0

for direc in [path_train, path_val]:
    filenames = sorted(os.listdir(direc+'input/'))
    for i, filename in enumerate(filenames):
        # import ipdb; ipdb.set_trace()
        image = imread(direc+'input/'+filename)
        if image.shape != (64,64,3):
            print(image.shape)
            continue
        image = image.reshape((1, 3, 64, 64))
        features[i+count, :, :, :] = image
        image2 = imread(direc+'target/'+filename).reshape((1, 3, 32, 32))
        targets[i+count, :, :, :] = image2
        if i % BATCH == 0:
            f.flush()
    count = i
    f.flush()
    print('{} set done'.format('Training' if direc == path_train else 'Validation'))
f.flush()
f.close()
