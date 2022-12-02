'''
MIT License

Copyright (c) 2022 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
from typing import Tuple, Union
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob

import tensorflow as tf

class TfdataPipeline:
    '''
    A class to create a tf.data.Dataset object from a directory of images.
    args:
        `dataset_dir`: str, the directory of the dataset
        `target_size`: list, the target size of the images (height, width, channels)
        `batch_size`: int, the batch size of the dataset
        `input_folder_name`: str, the name of the folder containing the input images
        `target_folder_name`: str, the name of the folder containing the target images
        `validation_split`: float, the percentage of the dataset to be used for validation
        `augmenting_list`: list, the list of augmenting methods to be used
        `augment_target_images`: bool, whether to augment the target images or not
    
    supported augmenting methods:
        random_crop: randomly crop the images
        flip_left_right: flip the images horizontally
        flip_up_down: flip the images vertically
        random_contrast: randomly change the contrast of the images
        random_saturation: randomly change the saturation of the images
        random_brightness: randomly change the brightness of the images
        random_hue: randomly change the hue of the images
        random_jpeg_quality: randomly change the jpeg quality of the images
    
    NOTE: Make your directory structure as follows:
        dataset_dir
            train
                input_folder_name
                    image1.jpg
                    image2.jpg
                    ...
                target_folder_name
                    image1.jpg
                    image2.jpg
                    ...
            test
                input_folder_name
                    image1.jpg
                    image2.jpg
                    ...
                target_folder_name
                    image1.jpg
                    image2.jpg
                    ...
    '''
    def __init__(
        self,
        dataset_dir: str,
        target_size: list               = [256, 256,3],
        batch_size: int                 = 16,
        input_folder_name: str          = 'blur',
        target_folder_name: str         = 'sharp',
        validation_split: float         = 0.1,
        augmenting_list: list           = ['random_crop', 'flip_left_right', 'flip_up_down', 'random_contrast', 'random_saturation'],
        augment_target_images: bool     = True,
    ) -> None:
        self.dataset_dir            = dataset_dir
        self.IMG_H                  = target_size[0]
        self.IMG_W                  = target_size[1]
        self.IMG_C                  = target_size[2]
        self.batch_size             = batch_size
        self.input_folder_name      = input_folder_name
        self.target_folder_name     = target_folder_name
        self.validation_split       = validation_split
        self.augmenting_list        = augmenting_list
        self.augment_target_images  = augment_target_images
        self.__dataset_type         = ['train', 'test', 'valid']
        self.random_ng              = tf.random.Generator.from_seed(12, alg='philox')

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                f'The directory {self.dataset_dir} does not exist.'
            )
        if not os.path.exists(os.path.join(self.dataset_dir, self.__dataset_type[0], self.input_folder_name)) or \
            not os.path.exists(os.path.join(self.dataset_dir, self.__dataset_type[0], self.target_folder_name)):

            raise FileNotFoundError(
                f'The directory {self.input_folder_name} or {self.target_folder_name} does not exist in train dataset.'
            )
        if not os.path.exists(os.path.join(self.dataset_dir, self.__dataset_type[1], self.input_folder_name)) or \
            not os.path.exists(os.path.join(self.dataset_dir, self.__dataset_type[1], self.target_folder_name)):
            raise FileNotFoundError(
                f"The directory {self.input_folder_name} or {self.target_folder_name} does not exist in test dataset."
            )
    
    def _load_train_val_image_file_names(self) -> tuple:
        '''
        Load the file names of the train dataset. Also split the train dataset into train and validation.
        '''
        
        input_image_files   = sorted(glob.glob(
            os.path.join(self.dataset_dir, self.__dataset_type[0], f'{self.input_folder_name}/*')
        ))

        target_image_files  = sorted(glob.glob(
            os.path.join(self.dataset_dir, self.__dataset_type[0], f'{self.target_folder_name}/*')
        ))

        train_input_files    = input_image_files[:int(len(input_image_files) * (1 - self.validation_split))]
        valid_input_files    = input_image_files[int(len(input_image_files) * (1 - self.validation_split)):]

        train_target_files   = target_image_files[:int(len(target_image_files) * (1 - self.validation_split))]
        valid_target_files   = target_image_files[int(len(target_image_files) * (1 - self.validation_split)):]

        return (train_input_files, train_target_files), (valid_input_files, valid_target_files)
    
    def _load_test_image_file_names(self) -> tuple:
        '''
        Load the file names of the test dataset.
        '''

        input_image_files   = sorted(glob.glob(
            os.path.join(self.dataset_dir, self.__dataset_type[1], f'{self.input_folder_name}/*')
        ))

        target_image_files  = sorted(glob.glob(
            os.path.join(self.dataset_dir, self.__dataset_type[1], f'{self.target_folder_name}/*')
        ))

        return input_image_files, target_image_files
    
    # A funtion to load the image files with docstring
    def load_input_target_image(self, input_image_path:str, target_image_path:str) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        Load the input and target images.
        args:
            `input_image_path`: str, the path of the content image
            `target_image_path`: str, the path of the style image
        return:
            `input_image`: tf.Tensor, the content image
            `target_image`: tf.Tensor, the style image
        '''
        # Read raw image from path
        input_image     = tf.io.read_file(input_image_path)
        target_image    = tf.io.read_file(target_image_path)

        # Decode the raw image
        input_image     = tf.io.decode_image(input_image, channels=self.IMG_C, expand_animations=False)
        target_image    = tf.io.decode_image(target_image, channels=self.IMG_C, expand_animations=False)

        # Change the dtype of the image to float32
        input_image     = tf.image.convert_image_dtype(input_image, tf.float32)
        target_image    = tf.image.convert_image_dtype(target_image, tf.float32)

        return input_image, target_image
    
    def _spatial_augment(self, x: tf.Tensor, seed:Union[tf.Tensor, tuple])-> tf.Tensor:
        '''
        Augment the images spatially.
        args:
            `x`:tf.Tensor, Any image/tensor
            `seed`: Union[tf.Tensor, tuple], the seed for the random number generator
        '''
        if 'random_crop' in self.augmenting_list:
            x = tf.image.stateless_random_crop(x, size=[self.IMG_H, self.IMG_W, self.IMG_C], seed=seed)

        if 'random_flip_left_right' in self.augmenting_list:
            x = tf.image.stateless_random_flip_left_right(x, seed=seed)

        if 'random_flip_up_down' in self.augmenting_list:
            x = tf.image.stateless_random_flip_up_down(x, seed=seed)

        if self.augmenting_list is None:
            pass
        
        return x

    def _color_augment(self, x: tf.Tensor, seed:Union[tf.Tensor, tuple])-> tf.Tensor:
        '''
        Augment the images color.
        args:
            `x`:tf.Tensor, Any image/tensor
            `seed`: Union[tf.Tensor, tuple], the seed for the random number generator
        '''
        if 'random_brightness' in self.augmenting_list:
            x = tf.image.stateless_random_brightness(x, max_delta=0.1, seed=seed)
        
        if 'random_contrast' in self.augmenting_list:
            x = tf.image.stateless_random_contrast(x, lower=0.2, upper=0.5, seed=seed)

        if 'random_saturation' in self.augmenting_list:
            x = tf.image.stateless_random_saturation(x, lower=0.2, upper=0.5, seed=seed)
        
        if 'random_hue' in self.augmenting_list:
            x = tf.image.stateless_random_hue(x, max_delta=0.2, seed=seed)
        
        if 'random_jpeg_quality' in self.augmenting_list:
            x = tf.image.stateless_random_jpeg_quality(x, 50, 100, seed=seed)

        if self.augmenting_list is None:
            return x
        
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
        
        return x
    
    def _augment_pairs(self, input_image_path: str, output_image: str)-> Tuple[tf.Tensor, tf.Tensor]:
        '''
        Augment input and target image with spatial augmentation and input image with color augmentation. 
        Augment target image with color augmentation if self.augment_target is True. 
        args:
            `input_image_path`: str, path to input image
            `output_image`: str, path to output image

        return:
            `input_image`: tf.Tensor, the augmented content image
            `target_image`: tf.Tensor, the augmented style image
        '''
        seed                        = self.random_ng.make_seeds(2)[0]

        input_image, target_image   = self.load_input_target_image(input_image_path, output_image)
        input_image                 = self._spatial_augment(input_image, seed)
        target_image                = self._spatial_augment(target_image, seed)

        input_image                 = self._color_augment(input_image, seed)
        if self.augment_target_images:
            target_image            = self._color_augment(target_image, seed)

        return input_image, target_image


    def _tf_dataset(self, input_image_paths: list, target_image_paths:list, do_augment:bool)-> tf.data.Dataset:
        '''
        Creates a tf.data.Dataset object from the input and target image paths
        args:
            `input_image_paths`:list, the list of input image paths
            `target_image_paths`:list, the list of target image paths
            `do_augment`:bool, whether to augment the images
        '''
        # Create a tf.data.Dataset object
        dataset = tf.data.Dataset.from_tensor_slices((input_image_paths, target_image_paths))

        # Map the function to load the input and target images
        if do_augment:
            dataset = dataset.map(self._augment_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(self.load_input_target_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = (dataset
                        .cache()
                        .shuffle(buffer_size=10)
                        .batch(self.batch_size)
                        .prefetch(buffer_size=tf.data.AUTOTUNE)
                )

        return dataset

    def data_loader(self, dataset_type:str = 'train', do_augment:bool = True)->tf.data.Dataset:
        '''
        Load the dataset.
        args:
            `dataset_type`:str, the type of the dataset, can be 'train' or 'test'
            `do_augment`:bool, whether to augment the images
        '''
        if dataset_type not in self.__dataset_type:
            raise ValueError(
                f'The dataset type {dataset_type} is not supported. '
                f'The supported dataset types are {self.__dataset_type}'
            )

        # Load the file names of the dataset
        if dataset_type == 'train':
            (train_input_image_paths, train_target_image_paths), _ = self._load_train_val_image_file_names()
            return self._tf_dataset(train_input_image_paths, train_target_image_paths, do_augment)
        
        elif dataset_type == 'valid':
            _, (valid_input_image_paths, valid_target_image_paths) = self._load_train_val_image_file_names()
            return self._tf_dataset(valid_input_image_paths, valid_target_image_paths, do_augment)
    
        else:
            input_image_paths, target_image_paths = self._load_test_image_file_names()
            return self._tf_dataset(input_image_paths, target_image_paths, do_augment)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    batch_size = 3
    # Test the class
    tfdataset = TfdataPipeline(dataset_dir='Dataset', batch_size=batch_size)

    ds = tfdataset.data_loader(dataset_type='train', do_augment=True)
    # ds = tfdataset.data_loader(dataset_type='valid', do_augment=True)
    # ds = tfdataset.data_loader(dataset_type='test', do_augment=True)

    for input_image, target_image in ds.take(3):
        tf.print(input_image.shape, target_image.shape)

        plt.figure(figsize=(10, 10))
        display_list = [input_image[1], target_image[1]]
        title = ['Input Image', 'Target Image']
        plt.subplot(1, 2, 1)
        plt.title(title[0])
        plt.imshow(display_list[0])
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title(title[1])
        plt.imshow(display_list[1])
        plt.axis('off')

    plt.show()