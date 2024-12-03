import os
import pickle

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_data(data_folder):
    df = pd.read_csv(f'{data_folder}/filenames_mapped_ages.csv')
    image_data = df[df['filename'].str.contains('.png', na=False)]

    train_dic = image_data[image_data['testing'] == 0].set_index('filename')['daysRemain'].to_dict()
    test_dic = image_data[image_data['testing'] == 1].set_index('filename')['daysRemain'].to_dict()

    image_size = (224, 224)
    batch_size = 32

    def preprocess(path):
        img = Image.open(path).resize(image_size)
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            img_array = np.stack([img_array]*3, axis=-1)
        img_array = img_array / 255.0
        return img_array
    
    train_images = []
    train_labels = []

    for image_id, days in tqdm(train_dic.items(), desc="Processing images"):
        img_path = os.path.join("../data/Images", image_id)
        if os.path.exists(img_path) and not np.isnan(days):
            image = preprocess(img_path)
            train_images.append(image)
            train_labels.append(days)
        else:
            pass

    test_images = []
    test_labels = []

    for image_id, days in tqdm(test_dic.items(), desc="Processing images"):
        img_path = os.path.join("../data/Images", image_id)
        if os.path.exists(img_path) and not np.isnan(days):
            image = preprocess(img_path)
            test_images.append(image)
            test_labels.append(days)
        else:
            pass

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return dict({'train_images': train_images, 'test_images': test_images, 'train_labels': train_labels, 'test_labels': test_labels})


def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)


if __name__ == '__main__':
    data_folder = '../data'
    create_pickle(data_folder)