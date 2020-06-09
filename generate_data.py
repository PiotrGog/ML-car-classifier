import os
from source.utils.data_augmentation import data_augmentation


if __name__ == "__main__":
    in_dir = str(os.sys.argv[1])
    out_dir = str(os.sys.argv[2])
    result_size = int(os.sys.argv[3])
    data_augmentation(in_dir, out_dir, result_size)
