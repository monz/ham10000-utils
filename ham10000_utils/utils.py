import pandas as pd
import random
import os
import shutil


def _split_file_extension(file_path):
    splits = os.path.splitext(file_path)
    if len(splits) == 2:
        path = splits[0]
        extension = splits[1]
    else:
        path = splits[0]
        extension = ''
    return extension, path


def _count_from_name(name, pattern):
    # check; not None nor empty
    assert name
    splits = name.rsplit(pattern)
    count = 0
    if len(splits) == 2:
        try:
            count = int(splits[1])
        except ValueError:
            print("Could not extract file count. Not a number.")  # todo: use logger, warning
    return count


def _duplicate_file_path(orig_file, count_sep='-'):
    file_ext, file_path = _split_file_extension(orig_file)
    file_count = _count_from_name(file_path, count_sep) + 1
    # if exists, remove previous file counter from file path
    if file_count > 1:
        file_path = file_path[:file_path.rfind(count_sep)]
    new_file = "{}{}{}{}".format(file_path, count_sep, file_count, file_ext)
    return new_file


# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
def _check_duplicate(filepath, allow_duplicates):
    """
    Checks whether the file path already exists and returns a new filepath when duplicates are allowed
    and the original filepath otherwise.

    Args:
        filepath (str): filepath to check
        allow_duplicates (bool): true to allow duplicates, false otherwise

    Returns:
        str: original filepath when filepath exists and duplicates are allowed, otherwise a new filepath for the
         duplicate is generated, e.g. by appending a file counter to the filename.
    """
    if os.path.exists(filepath) and allow_duplicates:
        filepath = _duplicate_file_path(filepath)
        #print("Duplicated filename: {}".format(filepath))  # todo: use logger, debug
        # recursively check for duplicates
        filepath = _check_duplicate(filepath, allow_duplicates)
    else:
        pass
    return filepath


def _copy(df_row, source, target, allow_duplicates=False):
    # check; not None nor empty
    assert source
    assert target

    # print("Would allow duplicates: {}".format(allow_duplicates))

    # create target directory
    if not os.path.exists(target):
        print(target)  # todo: use logger, debug
        os.makedirs(target)
    else:
        pass

    # prepare source and target file paths
    source_file = os.path.join(source, "{}.jpg".format(df_row['image_id']))
    target_file = os.path.join(target, "{}.jpg".format(df_row['image_id']))

    if os.path.exists(source_file):
        # create duplicate when allowed
        target_file = _check_duplicate(target_file, allow_duplicates)
        shutil.copyfile(source_file, target_file)
    else:
        raise Exception("Source file: '{}' does not exist.".format(source_file))  # todo: instead of throwing exception, log error/warning


def prepare_data_split(metadata_file_path, source_dir, target_dir, test_probability=.1):
    metadata = pd.read_csv(metadata_file_path)
    for _, row in metadata.iterrows():
        # get image target class
        image_class = row['dx']

        # assign to 'train' or 'test' group
        train_probability = 1-test_probability
        if random.choices([0, 1], (train_probability, test_probability), k=1)[0] == 0:
            # train data
            evaluated_target_dir = os.path.join(target_dir, "train", image_class)
        else:
            # test data
            evaluated_target_dir = os.path.join(target_dir, "test", image_class)

        _copy(row, source_dir, evaluated_target_dir)


def prepare_data(metadata_file_path, source_dir, target_dir, allow_duplicates=False):
    metadata = pd.read_csv(metadata_file_path)
    for _, row in metadata.iterrows():
        # get image target class
        image_class = row['dx']

        evaluated_target_dir = os.path.join(target_dir, image_class)

        _copy(row, source_dir, evaluated_target_dir, allow_duplicates)
