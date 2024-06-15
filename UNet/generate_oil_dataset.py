import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def save_img_color(img, path='.', form='png'):
    if not os.path.exists(path + '.' + form):
        fig, axs = plt.subplots(figsize=(img.shape[0], img.shape[1]), dpi=1)
        fig.subplots_adjust(0,0,1,1)
        axs.set_axis_off()
        axs.imshow(img, vmin=0, vmax=255)
        fig.savefig(path, format=form)
        plt.close(fig)

def process_img(args):
    in_folder, out_folder, im = args
    img = np.array(Image.open(os.path.join(in_folder, im)), dtype=np.uint8)
    index = 0
    for i in range(0, 620, 320):
        for j in range(0, 1240, 310):
            out_path = os.path.join(out_folder, im + '_' + str(index) + '.jpg')
            if not os.path.exists(out_path):
                ix = img[i:i+320, j:j+320, :]
                save_img_color(ix, out_folder+im+'_'+str(index), 'jpg')
            index += 1
    return im

def gen_img_320_overlap(in_folder, out_folder):
    files = os.listdir(in_folder)
    total_files = len(files)
    generated_files = len([name for name in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, name))])
    
    print(f"Total images to process: {total_files}")
    print(f"Images already generated: {generated_files}")

    with Pool(cpu_count()) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_img, [(in_folder, out_folder, im) for im in files])):
            generated_files += 1
            if generated_files % 10 == 0:
                print(f"{generated_files} images generated.")

def process_label(args):
    in_folder, out_folder, im = args
    img = np.array(Image.open(os.path.join(in_folder, im)), dtype=np.uint8)
    out_label = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint8)
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (img[i,j,:] == [51,221,225]).all():
                out_label[i,j,0] = 1
            if (img[i,j,:] == [255,0,124]).all():
                out_label[i,j,1] = 1
            if (img[i,j,:] == [255,0,124]).all():
                out_label[i,j,2] = 1
            if (img[i,j,:] == [153,76,0]).all():
                out_label[i,j,3] = 1
            if (img[i,j,:] == [255,204,51]).all():
                out_label[i,j,4] = 1

    out_path = os.path.join(out_folder, im + '.npy')
    if not os.path.exists(out_path):
        np.save(out_path, out_label)
    return im

def gen_label_5D(in_folder, out_folder):
    files = os.listdir(in_folder)
    total_files = len(files)
    generated_files = len([name for name in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, name))])
    
    print(f"Total labels to process: {total_files}")
    print(f"Labels already generated: {generated_files}")

    with Pool(cpu_count()) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_label, [(in_folder, out_folder, im) for im in files])):
            generated_files += 1
            if generated_files % 10 == 0:
                print(f"{generated_files} labels generated.")

def process_label_320_overlap(args):
    in_folder, out_folder, im = args
    img = np.load(os.path.join(in_folder, im))
    index = 0
    for i in range(0, 620, 320):
        for j in range(0, 1240, 310):
            out_path = os.path.join(out_folder, im.replace('.npy', '') + '_' + str(index) + '.npy')
            if not os.path.exists(out_path):
                ix = img[i:i+320, j:j+320, :]
                np.save(out_path, ix)
            index += 1
    return im

def gen_label_5D_320_overlap(in_folder, out_folder):
    files = os.listdir(in_folder)
    total_files = len(files)
    generated_files = len([name for name in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, name))])
    
    print(f"Total label tiles to process: {total_files}")
    print(f"Label tiles already generated: {generated_files}")

    with Pool(cpu_count()) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_label_320_overlap, [(in_folder, out_folder, im) for im in files])):
            generated_files += 1
            if generated_files % 10 == 0:
                print(f"{generated_files} label tiles generated.")

def process_flip_img(args):
    in_folder, out_folder, im = args
    img = np.array(Image.open(os.path.join(in_folder, im)), dtype=np.uint8)
    img_h = np.flip(img, 0)
    img_o = np.flip(img, 1)
    img_ho = np.flip(img)
    
    save_img_color(img, os.path.join(out_folder, im), form='jpg')
    save_img_color(img_h, os.path.join(out_folder, im + '_h'), form='jpg')
    save_img_color(img_o, os.path.join(out_folder, im + '_o'), form='jpg')
    save_img_color(img_ho, os.path.join(out_folder, im + '_ho'), form='jpg')
    return im

def gen_flip_img(in_folder, out_folder):
    files = os.listdir(in_folder)
    total_files = len(files)
    generated_files = len([name for name in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, name))])
    
    print(f"Total images to flip: {total_files}")
    print(f"Flipped images already generated: {generated_files}")

    with Pool(cpu_count()) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_flip_img, [(in_folder, out_folder, im) for im in files])):
            generated_files += 1
            if generated_files % 10 == 0:
                print(f"{generated_files} flipped images generated.")

def process_flip_label(args):
    in_folder, out_folder, im = args
    img = np.load(os.path.join(in_folder, im))
    img_h = np.flip(img, 0)
    img_o = np.flip(img, 1)
    img_ho = np.flip(img)
    
    np.save(os.path.join(out_folder, im), img)
    np.save(os.path.join(out_folder, im.replace('.npy', '_h')), img_h.astype(np.uint8))
    np.save(os.path.join(out_folder, im.replace('.npy', '_o')), img_o.astype(np.uint8))
    np.save(os.path.join(out_folder, im.replace('.npy', '_ho')), img_ho.astype(np.uint8))
    return im

def gen_flip_label(in_folder, out_folder):
    files = os.listdir(in_folder)
    total_files = len(files)
    generated_files = len([name for name in os.listdir(out_folder) if os.path.isfile(os.path.join(out_folder, name))])
    
    print(f"Total labels to flip: {total_files}")
    print(f"Flipped labels already generated: {generated_files}")

    with Pool(cpu_count()) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_flip_label, [(in_folder, out_folder, im) for im in files])):
            generated_files += 1
            if generated_files % 10 == 0:
                print(f"{generated_files} flipped labels generated.")

if __name__ == "__main__":
    print('Start...')

    augment_test_data = False
    
    ORIGINAL_DATASET_FOLDER       = './dataset/original_data/'
    PREPROCESSED_DATASET_FOLDER   = './dataset/'
    
    TRAIN_ORIGINAL_LABEL_PATH = '/content/neiron_oil/UNet/dataset/original_data/train/labels/'
    TRAIN_CONVERTED_LABEL_PATH    = PREPROCESSED_DATASET_FOLDER + 'train_label/'
    TRAIN_ORIGINAL_INPUT_PATH     = ORIGINAL_DATASET_FOLDER + 'train/images/'
    TRAIN_TILE_LABEL_PATH         = PREPROCESSED_DATASET_FOLDER + 'train_label_tile/'
    TRAIN_TILE_INPUT_PATH         = PREPROCESSED_DATASET_FOLDER + 'train_tile/'
    TRAIN_AUGMENTED_LABEL_PATH    = PREPROCESSED_DATASET_FOLDER + 'train_label_tile_aug/'
    TRAIN_AUGMENTED_INPUT_PATH    = PREPROCESSED_DATASET_FOLDER + 'train_tile_aug/'
    
    TEST_ORIGINAL_LABEL_PATH      = ORIGINAL_DATASET_FOLDER + 'test/labels/'
    TEST_CONVERTED_LABEL_PATH     = PREPROCESSED_DATASET_FOLDER + 'test_label/'
    TEST_ORIGINAL_INPUT_PATH      = ORIGINAL_DATASET_FOLDER + 'test/images/'
    TEST_TILE_LABEL_PATH          = PREPROCESSED_DATASET_FOLDER + 'test_label_tile/'
    TEST_TILE_INPUT_PATH          = PREPROCESSED_DATASET_FOLDER + 'test_tile/'
    if augment_test_data:
        TEST_AUGMENTED_LABEL_PATH     = PREPROCESSED_DATASET_FOLDER + 'test_label_tile_aug/'
        TEST_AUGMENTED_INPUT_PATH     = PREPROCESSED_DATASET_FOLDER + 'test_tile_aug/'
    
    try:
        os.makedirs(TRAIN_CONVERTED_LABEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_TILE_LABEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_TILE_INPUT_PATH, exist_ok=True)
        os.makedirs(TRAIN_AUGMENTED_LABEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_AUGMENTED_INPUT_PATH, exist_ok=True)

        os.makedirs(TEST_CONVERTED_LABEL_PATH, exist_ok=True)
        os.makedirs(TEST_TILE_LABEL_PATH, exist_ok=True)
        os.makedirs(TEST_TILE_INPUT_PATH, exist_ok=True)
        if augment_test_data:
            os.makedirs(TEST_AUGMENTED_LABEL_PATH, exist_ok=True)
            os.makedirs(TEST_AUGMENTED_INPUT_PATH, exist_ok=True)
    except Exception as ex:
        print(ex)
        exit(-1)
    
    gen_label_5D(TRAIN_ORIGINAL_LABEL_PATH, TRAIN_CONVERTED_LABEL_PATH)
    print('Gen train label 5D done.')
    gen_img_320_overlap(TRAIN_ORIGINAL_INPUT_PATH, TRAIN_TILE_INPUT_PATH)
    print('Gen train img 320 overlap done.')
    gen_label_5D_320_overlap(TRAIN_CONVERTED_LABEL_PATH, TRAIN_TILE_LABEL_PATH)
    print('Gen train label 320 overlap done.')
    gen_flip_img(TRAIN_TILE_INPUT_PATH, TRAIN_AUGMENTED_INPUT_PATH)
    print('Train img flip done.')
    gen_flip_label(TRAIN_TILE_LABEL_PATH, TRAIN_AUGMENTED_LABEL_PATH)
    print('Train label flip done.')
    
    gen_label_5D(TEST_ORIGINAL_LABEL_PATH, TEST_CONVERTED_LABEL_PATH)
    print('Gen test label 5D done.')
    gen_img_320_overlap(TEST_ORIGINAL_INPUT_PATH, TEST_TILE_INPUT_PATH)
    print('Gen test img 320 overlap done.')
    gen_label_5D_320_overlap(TEST_CONVERTED_LABEL_PATH, TEST_TILE_LABEL_PATH)
    print('Gen test label 320 overlap done.')
