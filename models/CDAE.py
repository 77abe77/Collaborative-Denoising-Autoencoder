import tensorflow as tf
from models import utils

class CDAE():
    def __init__(input_class, n_hidden=50, dropout_prob=0.2):
        self.sess = tf.Session()
        self.weight = self._init_variables() 
        self.sess.run(tf.initialize_all_variables())
        self.n_items = input_class.shape[1]
        self.n_users = input_class.shape[0]
        self.dropout_prob = dropout_prob
        self.LAMBDA = LAMBDA 
        self.optimizer = optimizer
        self.iteration = 0
        self.loss_fn = None #TODO
        self.transfer_fn = None #TODO
        self.output_fn = None #TODO

    def _set_up_training_ops(self):
        # Model Ops
        self.x_tilde = tf.placeholder(tf.float32, [None, self.n_items])
        self.hidden = lambda i: self.transfer_fn(tf.add_n( [tf.matmul(self.x_tilde, self.weight['W']), 
                                                            self.weight['V'][i], 
                                                            self.weight['b']]
                                                         ))
        self.reconstruction = lambda i: self.output_fn(tf.add(tf.matmul(self.hidden[i], tf.transpose(self.weight['W_prime'])),
                                                      self.weight['b_prime']))
        # Loss Function
        self.loss = tf.nn.l2_loss(tf.sub(self.x_tilde, self.reconstruction)

        # Gradient Ops
        #   - To my knowledge this when cost_gradient_apply gets called per user per item
        #     we are simulating online training. However, if we dont call apply_gradients
        #     we may be able to accumalate the gradients and apply them on a per user basis
        #     to simulate something like batch training.
        #   - TODO compare results of both online and batch methods
        
        self.dLoss_d = lambda var: self.optimizer.compute_gradients(self.loss, var_list=[var])[0][0]
        self.dCost_d = lambda var: tf.mul(1.0 / self.n_users, self.dLoss_d(var)) - tf.mul(self.LAMBDA, var)
        self.cost_gradient_apply = lambda var: self.optimizer.apply_gradients([[dCost_d(var), var]])
 
    def _create_augmented_O_set(self, ith_data, neg_sample_ratio = 5):
        pos_examples_indices = np.where(ith_data > 0)
        neg_examples_indices = np.where(ith_data == 0)
        aug_set = {index for index in pos_examples_indicies} 
        if len(neg_examples_indices) <= neg_sample_ratio * len(pos_examples_indices):
            for index in neg_examples_indicies:
                aug_set.add(index)
        else:
            for index in np.random.permutation(neg_examples)[0:neg_sample_ratio * len(pos_example_indices)]:
                aug_set.add(index)
        return aug_set

    def train(self):
        self._set_up_training_ops()
        while self.iteration < self.MAX_ITERATION or val_cost < val_cost_old: 
            val_cost_old = val_cost
            for i in xrange(self.n_users):
                ith_val_data = self.sess.run(tf.sparse_tensor_to_dense(val_data[i]))
                ith_train_data = self.sess.run(tf.sparse_tensor_to_dense(train_data[i]))
        
                x_tilde = self.sess.run(tf.nn.dropout(ith_train_data, self.dropout_prob))
                augmented_O_set = self._create_augmented_O_set(ith_train_data)
                for item in augmented_O_set:
                    self.sess.run(self.cost_gradient_apply(self.weight['W_prime'][item], feed_dict={self.x_tilde: x_tilde})
                    self.sess.run(self.cost_gradient_apply(self.weight['b_prime'][item], feed_dict={self.x_tilde: x_tilde})
                for item in np.where(x_tilde == 1):
                    self.sess.run(self.cost_gradient_apply(self.weight['W'][item], feed_dict={self.x_tilde: x_tilde})

                self.sess.run(self.cost_gradient_apply(self.weight['V'][i], feed_dict={self.x_tilde: x_tilde})
                self.sess.run(self.cost_gradient_apply(self.weight['b'], feed_dict={self.x_tilde: x_tilde})
                val_cost = self.sess.run(self.loss, feed_dict={self.x_tilde: ith_val_data})
                self.print_cost()
            self.iteration += 1
    
    def recommend(self, X, N=5):
        pass
    def _save_weights(self):
        pass
    def _load_weights(self):
        pass
    def print_weight(self):
        pass


    def _init_variables(self):
        all_weights = dict()
        all_weights['W'] = [tf.Variable(utils.xavier_init(1, self.n_hidden)), for _ in range(self.n_input)]
        all_weights['W_prime'] = [tf.Variable(utils.xavier_init(self.n_hidden, 1)), for _ in range(self.n_input)]
        all_weights['b'] = tf.Variable(tf.zeros([1, self.n_hidden], dtype=tf.float32))   
        all_weights['b_prime'] = [tf.Variable(tf.constant(0, dtype-tf.float32), dtype=tf.float32) for _ in range(self.n_users)]
        all_weights['V'] = [tf.Variable(tf.zeros([1, self.n_hidden])) for _ in range(self.n_users)] 

        return all_weights
