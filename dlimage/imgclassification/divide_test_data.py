import os
import random
import shutil


def divide_data_set(src_path, train_set_ratio=0.5, seed=100):
    if not os.path.isdir(src_path):
        return
    src_path = os.path.normpath(src_path)
    src_dir_name = os.path.basename(src_path)
    test_root = src_path.replace(src_dir_name, 'test_set')
    train_root = src_path.replace(src_dir_name, 'train_set')
    os.makedirs(test_root, exist_ok=True)
    os.makedirs(train_root, exist_ok=True)
    for root, dirs, files in os.walk(src_path):
        print(root, dirs, files)
        train_root = root.replace(src_dir_name, 'train_set')
        test_root = root.replace(src_dir_name, 'test_set')
        for d in dirs:
            os.makedirs(os.path.join(train_root, d), exist_ok=True)
            os.makedirs(os.path.join(test_root, d), exist_ok=True)
        random.Random(seed).shuffle(files)
        train_len = int(len(files) * train_set_ratio)
        test_len = int(len(files) * (1 - train_set_ratio))
        if train_len > 0:
            print(train_root)
            for f in files[0:train_len]:
                f = os.path.join(root, f)
                shutil.copy2(f, train_root)
        if test_len > 0:
            print(test_root)
            for f in files[train_len:]:
                f = os.path.join(root, f)
                shutil.copy2(f, test_root)


divide_data_set("data/dermatosis/dataset")
