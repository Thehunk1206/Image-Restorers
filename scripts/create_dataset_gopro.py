import os 
import sys
import shutil
from glob import glob

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Create train and test directories 
    os.mkdir(f'{path}/train/')
    os.mkdir(f'{path}/train/blur/')
    os.mkdir(f'{path}/train/sharp/') 

    os.mkdir(f'{path}/test/')
    os.mkdir(f'{path}/test/blur/')
    os.mkdir(f'{path}/test/sharp/')

    print('Directories created!')
    

def create_dataset(dataset_path, new_dataset_path):
    # Create directories
    create_dirs(new_dataset_path)

    # Train dataset
    for image_path in glob(os.path.join(dataset_path, 'train/*/blur/*')):
        print(f'Copying {image_path} to {new_dataset_path}/train/blur/')
        shutil.copy(image_path, f'{new_dataset_path}/train/blur/')
    
    for image_path in glob(os.path.join(dataset_path, 'train/*/sharp/*')):
        print(f'Copying {image_path} to {new_dataset_path}/train/sharp/')
        shutil.copy(image_path, f'{new_dataset_path}/train/sharp/')
    
    # Test dataset
    for image_path in glob(os.path.join(dataset_path, 'test/*/blur/*')):
        print(f'Copying {image_path} to {new_dataset_path}/test/blur/')
        shutil.copy(image_path, f'{new_dataset_path}/test/blur/')

    for image_path in glob(os.path.join(dataset_path, 'test/*/sharp/*')):
        print(f'Copying {image_path} to {new_dataset_path}/test/sharp/')
        shutil.copy(image_path, f'{new_dataset_path}/test/sharp/')

    print('Done!')

if __name__ == '__main__':
    dataset_path = sys.argv[1]
    new_dataset_path = sys.argv[2]

    create_dataset(dataset_path, new_dataset_path)