from argparse import ArgumentParser
from io import BytesIO
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm
import qoi
import zlib

TRAIN_IMG_PATH = 'train-images.idx3-ubyte'
TRAIN_LABEL_PATH = 'train-labels.idx1-ubyte'
TEST_IMG_PATH = 't10k-images.idx3-ubyte'
TEST_LABEL_PATH = 't10k-labels.idx1-ubyte'

TRAIN_NUM_SAMPLES = 60000
TEST_NUM_SAMPLES = 10000
# TRAIN_NUM_SAMPLES = 6
# TEST_NUM_SAMPLES = 100

num_workers = 16

parser = ArgumentParser()
parser.add_argument(
    '-k',
    type=int,
    default=2
)
parser.add_argument(
    '--format', '-f',
    type=str,
    choices=('png', 'qoi'),
    default='png'
)
parser.add_argument(
    '--concat-dim', '-d',
    type=int,
    choices=(0, 1),
    default=0,
    help='0 is vertical concatenation; 1 is horizontal.'
)
parser.add_argument(
    '--binary', '-b',
    action='store_true'
)
parser.add_argument(
    '--confusion-matrix', '--cm',
    action='store_true'
)
args = parser.parse_args()
print('Parameters:')
print('\n'.join([f'{k}:{v}' for k, v in vars(args).items()]))

INIT_K = args.k

USE_QOI = (args.format == 'qoi')

CONCAT_DIM = args.concat_dim

USE_BINARY_IMG = args.binary


def load_dataset(path, is_image, num_samples):
    sample_size = 28 * 28 if is_image else 1
    with open(path, 'rb') as f:
        if is_image:
            f.read(16) # header
        else:
            f.read(8)
        dataset = []
        for _ in range(num_samples):
            buf = f.read(sample_size)
            dataset.append(buf)
    return dataset


def ncd(arg):
    x1, C1, x2, C2 = arg

    if USE_QOI:
        a1 = qoi.decode(x1)
        a2 = qoi.decode(x2)
        a12 = np.concatenate((a1, a2), axis=CONCAT_DIM)
        C12 = len(zlib.compress(qoi.encode(a12)))
    else:
        bio = BytesIO()
        x12 = Image.new('L', (56, 28) if CONCAT_DIM == 0 else (28, 56), 'black')
        x12.paste(x1, (0, 0))
        x12.paste(x2, (28, 0) if CONCAT_DIM == 0 else (0, 28))
        x12.save(bio, 'png')
        C12 = len(bio.getvalue())

    # print(C1, C2, C12)
    # print((C12 - min(C1, C2)) / max(C1, C2))
    return (C12 - min(C1, C2)) / max(C1, C2)


def parallel(func, arg_list):
    with multiprocessing.Pool(num_workers) as p:
        res_list = p.map(func, arg_list)
    return res_list


def do_ncd(x1, train_images, train_image_C_list):
    if USE_QOI:
        C1 = len(zlib.compress(x1))
    else:
        bio = BytesIO()
        x1.save(bio, 'png')
        C1 = len(bio.getvalue())
    arg_list = list(zip(
        [x1] * TRAIN_NUM_SAMPLES,
        [C1] * TRAIN_NUM_SAMPLES,
        train_images,
        train_image_C_list
    ))
    return parallel(ncd, arg_list)


def do_knn(distances: list, labels):
    # print('len(distances)', len(distances))
    k = min(INIT_K, len(distances))
    while k > 0:
        sorted_index = np.argpartition(distances, k)[:k]
        top_k_labels = labels[sorted_index[:k]]
        # print('sorted_index', sorted_index)
        if k == 1:
            return top_k_labels[0]

        top_k_labels = list(top_k_labels)
        sorted_labels = sorted(set(top_k_labels), key=top_k_labels.count, reverse=True)
        # print('sorted_labels', sorted_labels)
        # print(list(map(top_k_labels.count, sorted_labels)))
        if len(sorted_labels) == 1:
            return sorted_labels[0]

        if top_k_labels.count(sorted_labels[0]) == top_k_labels.count(sorted_labels[1]):
            # if tie
            k -= 1
        else:
            return sorted_labels[0]


def buf2grayscale(buf):
    return np.frombuffer(buf, dtype=np.uint8).copy().reshape(28, 28, 1)


def buf2rgb(buf):
    arr = np.frombuffer(buf, dtype=np.uint8).copy().reshape(28, 28, 1)
    if USE_BINARY_IMG:
        arr[arr > 0] = 255
    return np.pad(arr, [(0, 0), (0, 0), (0, 2)], mode='constant')

def test(train_image):
    aa3 = buf2rgb(train_image)
    print(aa3.shape)
    qq = qoi.encode(aa3)
    print(qq)
    zqq = zlib.compress(qq)
    print(len(zqq))
    aa1 = buf2grayscale(train_image)
    for row in aa1:
        print(''.join([
            f'{p[0]:3d}'
            for p in row
        ]))
    exit(0)


def main():
    train_images = load_dataset(TRAIN_IMG_PATH, True, TRAIN_NUM_SAMPLES)
    train_labels = load_dataset(TRAIN_LABEL_PATH, False, TRAIN_NUM_SAMPLES)
    test_images = load_dataset(TEST_IMG_PATH, True, TEST_NUM_SAMPLES)
    test_labels = load_dataset(TEST_LABEL_PATH, False, TEST_NUM_SAMPLES)

    # test(train_images[0])

    pil_mode = '1' if USE_BINARY_IMG else 'L'

    train_images = [
        qoi.encode(buf2rgb(x))
        if USE_QOI else
        Image.frombytes(pil_mode, (28, 28), x)
        for x in train_images
    ]
    train_labels = np.array([
        int.from_bytes(y, 'little')
        for y in train_labels
    ])

    test_images = [
        qoi.encode(buf2rgb(x))
        if USE_QOI else
        Image.frombytes(pil_mode, (28, 28), x)
        for x in test_images
    ]
    test_labels = np.array([
        int.from_bytes(y, 'little')
        for y in test_labels
    ])
    # print(train_labels[0], test_labels[0])

    train_image_C_list = []
    for x2 in train_images:
        if USE_QOI:
            train_image_C_list.append(len(zlib.compress(x2)))
        else:
            bio = BytesIO()
            x2.save(bio, 'png')
            train_image_C_list.append(len(bio.getvalue()))

    predicted_labels = []
    tqdm_iter = tqdm(
        zip(test_images, test_labels),
        total=len(test_images),
        ncols=100,
        # disable=True
    )
    for x_img, x_label in tqdm_iter:
        all_ncd = do_ncd(x_img, train_images, train_image_C_list)
        predicted_labels.append(do_knn(all_ncd, train_labels))
        corrects = [
            1 if pred == true else 0
            for pred, true in zip(predicted_labels, test_labels)
        ]
        acc = sum(corrects) / len(corrects)
        tqdm_iter.set_description(f'Accuracy: {acc:.3f}')
        # print('predicted_label:', predicted_labels[-1])
        # print('true_label:', x_label)
        # print('---')

    if args.confusion_matrix:
        import seaborn
        import pandas as pd
        import matplotlib.pyplot as plt
        confusion_mat = np.zeros((10, 10), dtype=np.int32)
        for pred, true in zip(predicted_labels, test_labels):
            confusion_mat[pred, true] += 1
        cm_df = pd.DataFrame(confusion_mat)
        seaborn.heatmap(cm_df, annot=True)
        cm_name = f'cm/result_{args.format}{pil_mode}_k={INIT_K}_concat-dim={CONCAT_DIM}_acc={acc:.3f}.png'
        plt.savefig(cm_name)

    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)