from torchvision import datasets
import os


ROOT_DIR = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(ROOT_DIR)

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'Data/train_images')
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'Data/train.csv')
TEST_IMG_DIR = os.path.join(ROOT_DIR, 'Data/test_images')


def main():
    train_dataset = datasets.MNIST('./Data', train=True, download=True)
    test_dataset = datasets.MNIST('./Data', train=False, download=True)

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)

    with open(TRAIN_CSV_PATH, 'w') as fp:
        for idx, item in enumerate(train_dataset):
            basename = f'{idx:06d}'
            img_path = os.path.join(TRAIN_IMG_DIR, f'{basename}.jpg')

            # Save image and label
            item[0].save(img_path)
            fp.write(f'{basename}, {item[1]}\n')

    for idx, item in enumerate(test_dataset):
        basename = f'{idx:06d}'
        img_path = os.path.join(TEST_IMG_DIR, f'{basename}.jpg')

        # Save image
        item[0].save(img_path)


if __name__ == '__main__':
    main()
