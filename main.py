import tensorflow as tf
import numpy as np
from datasets import citeulikea
from datasets import citeulikeb
from datasets import utils
from models import CDAE

data = citeulikea.load_data()
training_set, test_set = utils.split_data(data)

model = CDAE(tf.AdaGradOptimizer())
model.train()
model




