import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_img_color(img, path='.', form='png'):
    fig, axs = plt.subplots(figsize=(img.shape[0], img.shape[1]), dpi=1)
    fig.subplots_adjust(0, 0, 1, 1)
    axs.set_axis_off()
    axs.imshow(img, vmin=0, vmax=255)
    fig.savefig(path, format=form)
    plt.close(fig)

def gen_img_320_overlap(in_folder, out_folder):
    files = os.listdir(in_folder)
    existing_files = set(os.listdir(out_folder))
    
    total_files = len(files)
    generated_files = 0
    
    for im in files:
        img = np.array(Image.open(in_folder + im), dtype=np.uint8)
        
        for i in range(0, 620, 320):
            for j in range(0, 1240, 310):
                index = f"{i}_{j}"
                file_name = f"{im}_{index}.jpg"
                if file_name not in existing_files:
                    ix = img[i:i+320, j:j+320, :]
                    save_img_color(ix, os.path.join(out_folder, file_name), 'jpg')
                    print(f"Generated and saved: {file_name}")
                    generated_files += 1
    
    print(f"Total images to generate: {total_files}")
    print(f"Images generated: {generated_files}")

def gen_label_5D(in_folder, out_folder):
    files = os.listdir(in_folder)
    existing_files = set(os.listdir(out_folder))
    
    total_files = len(files)
    generated_files = 0
    
    for im in files:
        file_name = f"{im}.npy"
        if file_name in existing_files:
            continue

        img = np.array(Image.open(os.path.join(in_folder, im)), dtype=np.uint8)
        out_label = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint8)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img[i, j, :] == [51, 221, 225]).all():
                    out_label[i, j, 0] = 1
                if (img[i, j, :] == [255, 0, 124]).all():
                    out_label[i, j, 1] = 1
                if (img[i, j, :] == [153, 76, 0]).all():
                    out_label[i, j, 3] = 1
                if (img[i, j, :] == [255, 204, 51]).all():
                    out_label[i, j, 4] = 1

        np.save(os.path.join(out_folder, file_name), out_label)
        print(f"Generated and saved: {file_name}")
        generated_files += 1

    print(f"Total labels to generate: {total_files}")
    print(f"Labels generated: {generated_files}")

def gen_label_5D_320_overlap(in_folder, out_folder):
    files = os.listdir(in_folder)
    existing_files = set(os.listdir(out_folder))
    
    total_files = len(files)
    generated_files = 0
    
    for im in files:
        img = np.load(os.path.join(in_folder, im))

        for i in range(0, 620, 320):
            for j in range(0, 1240, 310):
                index = f"{i}_{j}"
                file_name = f"{im.replace('.npy', '')}_{index}.npy"
                if file_name not in existing_files:
                    ix = img[i:i+320, j:j+320, :]
                    np.save(os.path.join(out_folder, file_name), ix)
                    print(f"Generated and saved: {file_name}")
                    generated_files += 1

    print(f"Total 5D labels to generate: {total_files}")
    print(f"5D labels generated: {generated_files}")

def gen_flip_img(in_folder, out_folder):
    files = os.listdir(in_folder)
    existing_files = set(os.listdir(out_folder))
    
    total_files = len(files)
    generated_files = 0
    
    for im in files:
        if any(
            [f"{im}{suffix}" in existing_files for suffix in ['', '_h', '_o', '_ho']]
        ):
            continue

        img = np.array(Image.open(os.path.join(in_folder, im)), dtype=np.uint8)
        img_h = np.zeros((img.shape), dtype=np.uint8)
        img_o = np.zeros((img.shape), dtype=np.uint8)
        img_ho = np.zeros((img.shape), dtype=np.uint8)
        for i in range(img.shape[-1]):
            img_h[..., i] = np.flip(img[..., i], 0)
            img_o[..., i] = np.flip(img[..., i], 1)
            img_ho[..., i] = np.flip(img[..., i])
        
        save_img_color(img, os.path.join(out_folder, im), form='jpg')
        save_img_color(img_h, os.path.join(out_folder, f"{im}_h"), form='jpg')
        save_img_color(img_o, os.path.join(out_folder, f"{im}_o"), form='jpg')
        save_img_color(img_ho, os.path.join(out_folder, f"{im}_ho"), form='jpg')
        print(f"Generated and saved: {im}, {im}_h, {im}_o, {im}_ho")
        generated_files += 4

    print(f"Total flipped images to generate: {total_files}")
    print(f"Flipped images generated: {generated_files}")

def gen_flip_label(in_folder, out_folder):
    files = os.listdir(in_folder)
    existing_files = set(os.listdir(out_folder))
    
    total_files = len(files)
    generated_files = 0
    
    for im in files:
        if any(
            [f"{im}{suffix}.npy" in existing_files for suffix in ['', '_h', '_o', '_ho']]
        ):
            continue

        img = np.load(os.path.join(in_folder, im))
        img_h = np.zeros(img.shape, dtype=np.uint8)
        img_o = np.zeros(img.shape, dtype=np.uint8)
        img_ho = np.zeros(img.shape, dtype=np.uint8)
        for i in range(img.shape[-1]):
            img_h[..., i] = np.flip(img[..., i], 0)
            img_o[..., i] = np.flip(img[..., i], 1)
            img_ho[..., i] = np.flip(img[..., i])
        
        np.save(os.path.join(out_folder, im), img)
        np.save(os.path.join(out_folder, f"{im.replace('.npy', '_h')}"), img_h)
        np.save(os.path.join(out_folder, f"{im.replace('.npy', '_o')}"), img_o)
        np.save(os.path.join(out_folder, f"{im.replace('.npy', '_ho')}"), img_ho)
        print(f"Generated and saved: {im}, {im}_h, {im}_o, {im}_ho")
        generated_files += 4

    print(f"Total flipped labels to generate: {total_files}")
    print(f"Flipped labels generated: {generated_files}")

if __name__ == "__main__":
    print('Start...')

    augment_test_data = False

    ORIGINAL_DATASET_FOLDER = './dataset/original_data/'
    PREPROCESSED_DATASET_FOLDER = './dataset/'

    TRAIN_ORIGINAL_LABEL_PATH = '/content/neiron_oil/UNet/dataset/original_data/train/labels/'
    TRAIN_CONVERTED_LABEL_PATH = PREPROCESSED_DATASET_FOLDER + 'train_label/'
    TRAIN_ORIGINAL_INPUT_PATH = ORIGINAL_DATASET_FOLDER + 'train/images/'
    TRAIN_TILE_LABEL_PATH = PREPROCESSED_DATASET_FOLDER + 'train_label_tile/'
    TRAIN_TILE_INPUT_PATH = PREPROCESSED_DATASET_FOLDER + 'train_tile/'
    TRAIN_AUGMENTED_LABEL_PATH = PREPROCESSED_DATASET_FOLDER + 'train_label_tile_aug/'
    TRAIN_AUGMENTED_INPUT_PATH = PREPROCESSED_DATASET_FOLDER + 'train_tile_aug/'

    TEST_ORIGINAL_LABEL_PATH = ORIGINAL_DATASET_FOLDER + 'test/labels/'
    TEST_CONVERTED_LABEL_PATH = PREPROCESSED_DATASET_FOLDER + 'test_label/'
    TEST_ORIGINAL_INPUT_PATH = ORIGINAL_DATASET_FOLDER + 'test/images/'
    TEST_TILE_LABEL_PATH = PREPROCESSED_DATASET_FOLDER + 'test_label_tile/'
    TEST_TILE_INPUT_PATH = PREPROCESSED_DATASET_FOLDER + 'test_tile/'
    if augment_test_data:
        TEST_AUGMENTED_LABEL_PATH = PREPROCESSED_DATASET_FOLDER + 'test_label_tile_aug/'
        TEST_AUGMENTED_INPUT_PATH = PREPROCESSED_DATASET_FOLDER + 'test_tile_aug/'

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
    if augment_test_data:
        gen_flip_img(TEST_TILE_INPUT_PATH, TEST_AUGMENTED_INPUT_PATH)
        print('Test img flip done.')
        gen_flip_label(TEST_TILE_LABEL_PATH, TEST_AUGMENTED_LABEL_PATH)
        print('Test label flip done.')

    try:
        shutil.rmtree(TRAIN_CONVERTED_LABEL_PATH, ignore_errors=True)
        shutil.rmtree(TRAIN_TILE_LABEL_PATH, ignore_errors=True)
        shutil.rmtree(TRAIN_TILE_INPUT_PATH, ignore_errors=True)

        shutil.rmtree(TEST_CONVERTED_LABEL_PATH, ignore_errors=True)
        if augment_test_data:
            shutil.rmtree(TEST_TILE_LABEL_PATH, ignore_errors=True)
            shutil.rmtree(TEST_TILE_INPUT_PATH, ignore_errors=True)
    except Exception as ex:
        print(ex)
        exit(-1)

    print("Done.")
