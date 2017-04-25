from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset
from scipy.misc import toimage


dataset_train = H5PYDataset('/Tmp/alitaiga/ift6266/coco_cropped.h5', which_sets=('train',))
train_stream = DataStream(
    dataset_train,
    iteration_scheme=ShuffledScheme(dataset_train.num_examples, 1)
)
iterator = train_stream.get_epoch_iterator()
center = (32,32)

for batch in iterator:
    print(batch)
    input = batch[0][:,:,:,:].reshape((1,64,64,3))
    input = input[0,:,:,:]
    # toimage(input).show()
    # Image.fromarray(input, 'RGB').show()
    # plt.show()
    target = batch[1][0,:,:,:].reshape((32,32,3))
    # plt.imshow(target)
    full = input
    full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = target
    # import ipdb; ipdb.set_trace()
    toimage(full).show()
    # plt.show()
    import ipdb; ipdb.set_trace()
