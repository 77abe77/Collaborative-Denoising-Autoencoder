import tensorflow as tf
import numpy as np

class AbstractInputSource():
    FILE_PATH = None
    class Data():
        def __init__(self, train_data, test_data, full_data):
            self.ith_train_data = train_data
            self.ith_test_data = test_data
            self.ith_full_data = full_data
        def get_test_neg_indices(self):
            return [neg_item for neg_item in xrange(self.n_items) if neg_item not in self.ith_test_data]
        def get_train_neg_indices(self):
            return [neg_item for neg_item in xrange(self.n_items) if neg_item not in self.ith_train_data]
        def get_test_data(self):
            return self.ith_test_data
        def get_train_data(self):
            return self.ith_train_data
        def get_full_data(self):
            return self.ith_full_data

    def __init__(self, sparse=True):
        if self.FILE_PATH == None:
            raise NotImplementedError
        self.data = []
        self.n_items = self._calculate_item_count()
        self.load_data_as_sparse() if sparse else self.load_data()
        self.n_users = len(self.data)

    def _make_sparse_from_raw_set(self, raw_set):
        indices = [[0, i] for i in raw_set]
        values = [1] * len(indices)
        shape = [1, self.n_items]
        return tf.SparseTensor(indices=indices, values=values, shape=shape)

    def get_data(self):
        return self.data

    def new_train_set(self):
        self.load_data_as_sparse() if self.sparse else self.load_data()  

    def rand_split_data(self, data, ratio=0.8):
        rand_ints = np.random.randint(1, data.size, data.size)
        train_indices = np.where(rand_ints <= ratio * data.size)[0]
        test_indices = np.where(rand_ints > ratio * data.size)[0]
        return data[train_indices], data[test_indices] 

    def _calculate_item_count(self):
        raise NotImplementedError

    def load_data_as_sparse(self):
        raise NotImplementedError 

    def load_data(self):
        raise NotImplementedError 

    def get_n_items(self):
        return self.n_items

    def get_n_users(self):
        return self.n_users 
