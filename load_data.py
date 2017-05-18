import pickle as pk
import numpy as np

# ref: tensorflow.examples.tutorials.mnist.input_data

# helper function for loading data
def load_data(name):
    print('Loading data... ', end='')
    if name == 'euc-kr':
        file_path = 'dataset/euc-kr.pkl'
    else:
        print('No such Data!: %s' % name)
        raise FileNotFoundError

    with open(file_path, 'rb') as f:
        data = pk.load(f)
        # train = DataSet(data['train_images'], data['train_labels'])
        # test = DataSet(data['test_images'], data['test_labels'])
        train = DataSet(data['images'], data['labels_cho'])
        test = None
        print('done!')
        return Datasets(train=train, test=test)


class DataSet:

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        assert images.shape[0] == labels.shape[0]
        self.num_examples = images.shape[0]
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        assert 0 <= batch_size <= self.num_examples

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.num_examples)
            np.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start
            images_rest_part = self.images[start:self.num_examples]
            labels_rest_part = self.labels[start:self.num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.num_examples)
                np.random.shuffle(perm)
                self.images = self.images[perm]
                self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.images[start:end], self.labels[start:end]


class Datasets:

    def __init__(self, train, test):
        self.train = train
        self.test = test
        # assert train.images.shape[1] == test.images.shape[1]
        # assert train.labels.shape[1] == test.labels.shape[1]
        # self.image_size = train.images.shape[1]
        # self.label_size = train.labels.shape[1]