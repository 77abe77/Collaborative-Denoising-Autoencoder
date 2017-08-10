import os
from datasets.AbstractInputSource import AbstractInputSource
import numpy as np

class CiteYouLikeA(AbstractInputSource):
    FILE_PATH = '/home/abemillan/Developer/CDAE_ML/datasets/citeulike-a/users.dat'# TODO make this path always work: os.path.abs(__file__)
    def _loop_through_file_with_fn(self, fn):
        # TODO
        pass
    def _calculate_item_count(self, n_items=0):
        with open(self.FILE_PATH, 'rb') as input_file:
            for user in input_file:
                user = user.strip()
                user = user.split()[1:]
                for item in user:
                    if int(item) > n_items:
                        n_items = int(item)
        return (n_items + 1)
 
    def load_data_as_sparse(self):
        with open(self.FILE_PATH, 'rb') as input_file:
            for user in input_file:
                user = user.strip()
                user = user.split()[1:]
                full_items = np.array([int(item) for item in user])
                train_indices, test_indices = self.rand_split_data(full_items)
                train_data = self._make_sparse_from_raw_set(train_indices)
                test_data = test_indices
                full_data = self._make_sparse_from_raw_set(full_items)
                
                self.data.append(self.Data(train_data, test_data, full_data))
                 
    def load_data(self):
        with open(self.FILE_PATH, 'rb') as input_file:
             for user in input_file:
                user = user.strip()
                user = user.split()[1:]
                full_items = np.array([int(item) for item in user])
                train_items, test_items = self.rand_split_data(full_items)
                train_data = np.array([1.0 if item in train_items else 0.0 for item in xrange(self.n_items)])
                test_data = test_items
                full_data = np.array([1.0 if item in full_items else 0.0 for item in xrange(self.n_items)])
                 
                self.data.append(self.Data(train_data, test_data, full_data))

