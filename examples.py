eimport os
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
#from skimage.transform import resize


def resize_mscoco():
    '''
    function used to create the dataset,
    Resize original MS_COCO Image into 64x64 images
    '''

    # PATH need to be fixed
    data_path="/Users/Adrien/Repositories/IFT6266h17/inpainting/val2014"
    save_dir = "/Users/Adrien/Repositories/IFT6266h17/dataset/val/input/"
    save_dir2 = "/Users/Adrien/Repositories/IFT6266h17/dataset/val/target/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir2)

    preserve_ratio = True
    image_size = (64, 64)
    #crop_size = (32, 32)

    imgs = glob.glob(data_path+"/*.jpg")


    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print i, len(imgs), img_path

        img_array = np.array(img)

        #cap_id = os.path.basename(img_path)[:-4]

        # Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            continue
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
        

        input = Image.fromarray(input)
        target = Image.fromarray(target)
        input.save(save_dir + os.path.basename(img_path))
        target.save(save_dir2 + os.path.basename(img_path))


def show_examples(batch_idx, batch_size,
                  mscoco="/Users/Adrien/Repositories/IFT6266h17/inpainting/", split="train2014", caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        # Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]


        #Image.fromarray(img_array).show()
        Image.fromarray(input).show()
        Image.fromarray(target).show()
        print i, caption_dict[cap_id]



if __name__ == '__main__':
    resize_mscoco()
    #show_examples(5, 10)
