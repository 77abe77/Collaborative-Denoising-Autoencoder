import tensorflow as tf
from models import CDAE

class CDAETesting(tf.test.TestCase):
    def setUp(self):
        self.CDAE_test = CDAE()
    def test_variable_packing(self):
        X = tf.constant()
        with
    def test_presision(self):
        test_set_positive_indicies = np.array([55, 32, 89, 49])
        model_result = np.random.random_sample(100)
    def test_recall(self):
        pass
    def test_MAP(self):
        pass
    
if __name__ == '__main__':
    tf.test.main()
