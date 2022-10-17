import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb_for_dacon():
    """Create lmdb files for DACON challenge train dataset.

    Usage:
        * dacon/train/lr
        * dacon/train/hr
    """
    # HR images
    folder_path = 'dacon/train/hr'
    lmdb_path = 'dacon/train/hr.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LR images
    folder_path = 'dacon/train/lr'
    lmdb_path = 'dacon/train/lr.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_for_seoul():
    """Create lmdb files for DACON challenge train dataset.

    Usage:
        * seoul/hr
    """
    # HR images
    folder_path = 'seoul/hr'
    lmdb_path = 'seoul/hr.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        help=("Options: 'dacon', 'seoul' You may need to modify the corresponding configurations in codes."))
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset == 'dacon':
        create_lmdb_for_dacon()
    elif dataset == 'seoul':
        create_lmdb_for_seoul()
    else:
        raise ValueError('Wrong dataset.')
