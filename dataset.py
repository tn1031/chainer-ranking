import numpy as np
import chainer
from chainercv import transforms


def _transform(img, is_train):
    img = img.transpose(2, 0, 1).astype(np.float32)
    if is_train:
        img = transforms.random_flip(img, x_random=True)
    img = transforms.resize(img, (128, 128))
    img[0, :] = img[0, :] / 127.5 - 1
    img[1, :] = img[1, :] / 127.5 - 1
    img[2, :] = img[2, :] / 127.5 - 1
    return img


class SimpleIndexImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        x = self.images[i]
        return i, _transform(x, False)


class HipsterWarsImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, ratings, images, is_train):
        self.ratings = ratings
        self.images = images
        self.is_train = is_train

    def __len__(self):
        return len(self.ratings)

    def get_example(self, idx):
        i, j, r = self.ratings[idx]
        xi, xj = self.images[int(i)], self.images[int(j)]
        xi = _transform(xi, self.is_train)
        xj = _transform(xj, self.is_train)
        return xi, xj, np.array(r, dtype=np.int32)


class HipsterWarsDataset(chainer.dataset.DatasetMixin):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def get_example(self, i):
        return tuple(map(int, self.ratings[i]))