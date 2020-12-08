import pandas as pd
import random
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing


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


def show_image_sizes(images_root_dir):
    image_sizes = set()
    for dirpath, dir_names, file_names in os.walk(images_root_dir):
        for file in file_names:
            img_path = os.path.join(dirpath, file)
            img = tf.io.read_file(img_path)
            img = tf.io.decode_jpeg(img)
            image_sizes.add(tuple(img.shape.as_list()))

    for size in image_sizes:
        print(size)


class ImageDatasetGenerator:
    def __init__(self,
                 data_generator: keras.preprocessing.image.ImageDataGenerator,
                 image_dir,
                 image_height,
                 image_width,
                 image_channels,
                 batch_size,
                 seed,
                 data_type,
                 label_type,
                 image_scale_factor=1,):
        self.data_generator = data_generator
        self.img_dir = image_dir
        self.img_height = image_height
        self.img_width = image_width
        self.img_scale_factor = image_scale_factor
        self.img_channels = image_channels
        self.batch_size = batch_size
        self.seed = seed
        self.data_type = data_type
        self.label_type = label_type

    def _image_iter(self, subset):
        return lambda: self.data_generator.flow_from_directory(
                directory=self.img_dir,
                seed=self.seed,
                target_size=(int(self.img_height*self.img_scale_factor),
                             int(self.img_width*self.img_scale_factor)),
                batch_size=self.batch_size,
                subset=subset,
                shuffle=False,
                # save_prefix='img',
                # save_to_dir='/home/markus/data/dataset/HAM10000/foo',
                # save_format='jpg',
        )

    def generate(self, subset):
        image_iter = self._image_iter(subset)
        data_count = image_iter().samples
        label_count = image_iter().num_classes
        return tf.data.Dataset.from_generator(
            image_iter,
            output_types=(self.data_type, self.label_type),
            output_shapes=(
                [None,
                 int(self.img_height*self.img_scale_factor),
                 int(self.img_width*self.img_scale_factor),
                 self.img_channels],
                [None, label_count]
            ),  # 'None' for unknown/variable batch size
        ), data_count, label_count

    def data_size(self):
        return int(self.img_height*self.img_scale_factor), int(self.img_width*self.img_scale_factor), self.img_channels


# https://www.tensorflow.org/guide/keras/preprocessing_layers
# https://www.tensorflow.org/tutorials/images/data_augmentation
# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal_and_vertical"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.1),
])


# input size should be ordered (height, width, channels)
def get_model(input_size, output_size):
    model = keras.models.Sequential([
        keras.Input(shape=input_size),
        data_augmentation,
        keras.layers.Conv2D(filters=32, kernel_size=11, padding='valid', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=64, kernel_size=11, padding='valid', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=128, kernel_size=9, padding='same', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=128, kernel_size=9, padding='same', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=254, kernel_size=9, padding='same', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(192, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_size),
    ])

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy'],
    )

    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_dir = '/home/markus/data/dataset/HAM10000/'
    metadata_file = os.path.join(base_dir, 'HAM10000_metadata.csv')
    source_dir = os.path.join(base_dir, 'workdir')
    target_dir = os.path.join(base_dir, 'sorted-part_1')

    # may have used the following link, instead of ImageDataGenerator
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/folder_dataset/ImageFolder

    # prepare data structure for tensorflow directory iterator
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/DirectoryIterator
    # prepare_data_split(metadata_file, source_dir, target_dir, test_probability=0.1)
    # prepare_data(metadata_file, source_dir, target_dir)

    # show image shapes
    # show_image_sizes(target_dir)

    # test usage of tensorflow directory iterator
    target_dir_train = os.path.join(target_dir, 'train')
    target_dir_test = os.path.join(target_dir, 'test')

    image_data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
    dataset_generator = ImageDatasetGenerator(
        data_generator=image_data_generator,
        image_dir=target_dir,
        image_height=450,
        image_width=600,
        image_channels=3,
        batch_size=64,
        seed=0,
        data_type=tf.int8,
        label_type=tf.int8,
    )

    # split data into train, validation data
    # https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
    train_dataset, train_data_count, train_label_count = dataset_generator.generate("training")
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset, test_data_count, test_label_count = dataset_generator.generate("validation")
    # test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # print(test_dataset.cardinality() == tf.data.UNKNOWN_CARDINALITY)
    # print(test_dataset.cardinality() == tf.data.INFINITE_CARDINALITY)

    print(train_data_count, train_label_count)
    print(test_data_count, test_label_count)
    print(type(dataset_generator.data_size()))
    print(dataset_generator.batch_size)

    # print(dataset.element_spec)
    # print(dataset)
    for image, label in test_dataset.take(1):
        print(image.shape)

        # print(image.shape)
        # print(label.shape)
        # break  # take only the first batch



