# https://github.com/facebookresearch/mmbt/blob/master/scripts/food_101.py
import os

from tqdm import tqdm
import re

from os.path import join
import json
from nltk.tokenize import word_tokenize
from collections import Counter

CLASS_NAME = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
              'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli',
              'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry',
              'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
              'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
              'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
              'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
              'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
              'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
              'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
              'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
              'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
              'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
              'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
              'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
TEXT_MAX_LENGTH = 100
MIN_FREQ = 3
NUMBER_OF_SAMPLES_PER_CLASS = None

def format_food101_dataset(dataset_root_path):
    print("Parsing data...")
    data = parse_data(dataset_root_path)
    print("Saving everything into format...")
    save_in_format(data, dataset_root_path)


def get_tokens(text_file_path):
    content = open(text_file_path, 'r').read()
    for c in '<>/\\+=-_[]{}\'\";:.,()*&^%$#@!~`|':
        content = content.replace(c, ' ')
    content = re.sub("\s\s+", ' ', content)
    content = content.lower().replace("\n", " ")

    text_tokens = word_tokenize(content)
    if 'ingredients' in text_tokens:
        ingredients_index = 0
        for token in text_tokens:
            if "ingredients" in token:
                ingredients_index = text_tokens.index(token)
        text_tokens = text_tokens[ingredients_index + 1: ingredients_index + 1 + TEXT_MAX_LENGTH]
        return text_tokens
    else:
        return None


def parse_data(source_dir):
    splits = ["train", "test"]
    data = {split: [] for split in splits}

    token_counter = Counter()
    label_counter = Counter()
    for split in splits:
        for label in tqdm(os.listdir(join(source_dir, "images", split))):
            label = str(label)
            for img in os.listdir(join(source_dir, "images", split, label)):
                match = re.search(
                    r"(?P<name>\w+)_(?P<num>[\d-]+)\.(?P<ext>\w+)", img
                )
                num = match.group("num")
                dobj = {}
                dobj["id"] = label + "_" + img
                dobj["label"] = label

                if split == "train" and NUMBER_OF_SAMPLES_PER_CLASS is not None and label_counter[
                    label] >= NUMBER_OF_SAMPLES_PER_CLASS:
                    continue
                else:
                    txt_path = join(
                        source_dir, "texts_txt", label, "{}_{}.txt".format(label, num)
                    )
                    if not os.path.exists(txt_path):
                        continue
                    dobj["text_path"] = txt_path
                    text_tokens = get_tokens(txt_path)
                    if text_tokens is None:
                        continue
                    dobj["text_tokens"] = text_tokens
                    dobj["img_path"] = join(source_dir, "images", split, label, img)
                    if not os.path.exists(dobj["img_path"]):
                        continue
                    if split == "train":
                        # only update tokens in training dataset
                        token_counter.update(text_tokens)

                    data[split].append(dobj)

                    label_counter.update([label])

    return data


def save_in_format(data, target_path):
    """
    Stores the data to @target_dir. It does not store metadata.
    """

    for split_name in data:
        print(split_name)
        jsonl_loc = join(target_path, split_name + ".json")
        with open(jsonl_loc, "w") as jsonl:
            json.dump(data[split_name], jsonl)


if __name__ == "__main__":
    dataset_dir = "/workspace/Dataset/UPMC_Food101"
    format_food101_dataset(dataset_dir)