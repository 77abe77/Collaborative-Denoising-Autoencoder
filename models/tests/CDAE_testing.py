import tensorflow as tf
from models import CDAE

class CDAETesting(tf.test.TestCase):
    def setUp(self):
        self.CDAE_test = CDAE()
    def test_variable_packing(self):
        X = tf.constant()
        with

if __name__ == '__main__':
    tf.test.main()
