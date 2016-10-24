import tensorflow as tf
import numpy as np
from models import utils
import json

class CDAE():
    def __init__(self, input_class, sparse=True):
        # Input Data Properties
        print 'Importing Data ...'
        self.input_data = input_class(sparse=sparse)
        print 'Done.'
        self.sparse = sparse
        self.n_items = self.input_data.get_n_items()
        self.n_users = self.input_data.get_n_users()
        
    def _set_up_training_ops(self, **kwargs):
        '''
        '''
        print 'Set up Training Operations...'
        # Init parameters
        dropout_prob = kwargs.get('dropout_prob') if kwargs.get('dropout_prob') else 0.2
        transfer_fn = kwargs.get('transfer_fn') if kwargs.get('transfer_fn') else tf.sigmoid
        output_fn = kwargs.get('output_fn') if kwargs.get('output_fn') else tf.identity
        n_hidden = kwargs.get('n_hidden') if kwargs.get('n_hidden') else 50
        
        # AdaGrad Parameters
        ada_beta = kwargs.get('ada_beta') if kwargs.get('ada_beta') else 1.0
        ada_learning_rate = kwargs.get('ada_learning_rate') if kwargs.get('ada_learning_rate') else 0.01
        optimizer = tf.train.AdagradOptimizer(ada_learning_rate, ada_beta)
        #optimizer = tf.train.GradientDescentOptimizer(ada_learning_rate)

        # Regularization Parameter
        reg_lambda = kwargs.get('reg_lambda') if kwargs.get('reg_lambda') else 1.0

        # Model Ops
        self.prime_indices = tf.placeholder(tf.int32, [None])
        self.W_indices = tf.placeholder(tf.int32, [None])
        self.W_prime_coords = tf.placeholder(tf.int32, [None, None])
        self.b_prime_coords = tf.placeholder(tf.int32, [None, None])
        self.W_coords = tf.placeholder(tf.int32, [None, None])
        self.V_coords = tf.placeholder(tf.int32, [None, None])

        self.y = tf.placeholder(tf.float32, [None, self.n_items])
        self.z = transfer_fn(tf.add_n( [tf.matmul(self.y, self.weight['W']), 
                                                  tf.reshape(self.weight['V_u'], [1, n_hidden]), 
                                                  tf.reshape(self.weight['b'], [1, n_hidden])]
                                                ))
        self.y_hat = output_fn(tf.add(tf.matmul(self.z, tf.transpose(self.weight['W_prime'])),
                                                   self.weight['b_prime'])
                              )   
        # Loss Function
        # TODO - Figure out how to include logistic loss efficiently
        self.loss = tf.nn.l2_loss(tf.sub(self.y, self.y_hat))

        # Gradient Ops
        #   - To my knowledge this when cost_gradient_apply gets called per user per item
        #     we are simulating online training. However, if we dont call apply_gradients
        #     we may be able to accumalate the gradients and apply them on a per user basis
        #     to simulate something like batch training.
        #   - TODO compare results of both online and batch methods
        
        dloss_dweights = optimizer.compute_gradients(self.loss, var_list=[self.weight['W_prime'],
                                                                          self.weight['b_prime'],
                                                                          self.weight['W'],
                                                                          self.weight['V'],
                                                                          self.weight['b']])
        # Add l2 Regularization Penalty
        W_prime_grads = tf.gather(dloss_dweights[0][0], self.prime_indices) + reg_lambda * tf.gather(self.weight['W_prime'], self.prime_indices)
        b_prime_grads = tf.gather(tf.transpose(dloss_dweights[1][0]), self.prime_indices) + reg_lambda * tf.gather(tf.transpose(self.weight['b_prime']), self.prime_indices)
        W_grads = tf.gather(dloss_dweights[2][0], self.W_indices) + reg_lambda * tf.gather(self.weight['W'], self.W_indices)
        V_grads = tf.gather(dloss_dweights[3][0], self.user_index) + reg_lambda * tf.gather(self.weight['V'], self.user_index)
        b_grads = dloss_dweights[4][0] + reg_lambda * self.weight['b']

        W_prime_grads = self._make_dense_grad(W_prime_grads, [self.n_items, n_hidden], self.W_prime_coords)
        b_prime_grads = self._make_dense_grad(b_prime_grads, [1, self.n_items], self.b_prime_coords)
        W_grads = self._make_dense_grad(W_grads, [self.n_items, n_hidden], self.W_coords)
        V_grads = self._make_dense_grad(V_grads, [self.n_users, n_hidden], self.V_coords)

        
 
        self._train = optimizer.apply_gradients([ [W_prime_grads, self.weight['W_prime']],
                                                  [b_prime_grads, self.weight['b_prime']],
                                                  [W_grads, self.weight['W']],
                                                  [b_grads, self.weight['b']],
                                                  [V_grads, self.weight['V']] ])

        self.init_op = tf.initialize_all_variables()
        print 'Done.'

    def _make_dense_grad(self, grad, shape, coords):
        values = tf.reshape(grad, [-1])
        return tf.sparse_to_dense(sparse_indices=coords, output_shape=shape, sparse_values=values)
 
    def _create_augmented_O_set(self, ith_data, neg_sample_ratio = 5):
        '''
        '''
        pos_examples_indices = np.where(ith_data > 0)[0]
        neg_examples_indices = np.where(ith_data == 0)[0]
        aug_set = {index for index in pos_examples_indices} 
        if len(neg_examples_indices) <= neg_sample_ratio * len(pos_examples_indices):
            for index in neg_examples_indices:
                aug_set.add(index)
        else:
            for index in np.random.permutation(neg_examples_indices)[0:neg_sample_ratio * len(pos_examples_indices)]:
                aug_set.add(index)
        return sorted(list(aug_set))

    def train(self, **kwargs):
        '''
            Method  
        '''
        MAX_ITERATION = kwargs.get('max_iteration') if kwargs.get('max_iteration') else 1000
        dropout_prob = kwargs.get('dropout_prob') if kwargs.get('dropout_prob') else 0.2
        n_hidden = kwargs.get('n_hidden') if kwargs.get('n_hidden') else 50
        dropout_prob = tf.constant(dropout_prob, dtype=tf.float32)
        iteration = 0

        self.weight = self._create_variables(n_hidden) 
        self._set_up_training_ops(**kwargs)

        old_avg_cost = None
        model_improved = True
        input_data = self.input_data.get_data()
        sess = tf.Session()
        print 'Initialing all Variables...'
        sess.run(self.init_op)
        print 'Done.'
        print 'Start Training... '
        while iteration < MAX_ITERATION or model_improved: 
            old_avg_cost = avg_cost if iteration >= 1 else None 
            sum_cost = 0
            for u in xrange(self.n_users):
                val_data = [input_data[u].get_full_data()]
                train_data = input_data[u].get_train_data()
                
                if self.sparse:
                    try:
                        val_data = sess.run(tf.sparse_tensor_to_dense(val_data))
                        train_data = sess.run(tf.sparse_tensor_to_dense(train_data))
                    except:
                        print 'user: {}'.format(u)
                y_tilde = sess.run(tf.nn.dropout(tf.constant(train_data, dtype=tf.float32, shape=[1, self.n_items]), dropout_prob))
                augmented_O_set = self._create_augmented_O_set(np.array(val_data))
                observed_O_set = sorted(list(np.where(y_tilde > 0)[1])) 
                
                print 'y_tilde: {}'.format(y_tilde)
                print ''
                print 'user_index: {}'.format(u)
                print ''
                print 'prime_indices: {}'.format(augmented_O_set)
                print ''
                print 'W_indices: {}'.format(observed_O_set)
                print ''
                print 'W_prime_coords: {}'.format([ [i, j] for i in augmented_O_set for j in xrange(n_hidden)])
                print ''
                print 'b_prime_coords: {}'.format([ [0, i] for i in augmented_O_set])
                print ''
                print 'W_coords: {}'.format([ [i, j] for i in observed_O_set for j in xrange(n_hidden)])
                print ''
                print 'V_coords: {}'.format([ [u, i] for i in xrange(n_hidden)])

                feed_dict = {self.y: y_tilde,
                             self.user_index: u,
                             self.prime_indices: augmented_O_set,
                             self.W_indices: observed_O_set, 
                             self.W_prime_coords: [ [i, j] for i in augmented_O_set for j in xrange(n_hidden)],
                             self.b_prime_coords: [ [0, i] for i in augmented_O_set],
                             self.W_coords: [ [i, j] for i in observed_O_set for j in xrange(n_hidden)],
                             self.V_coords: [ [u, i] for i in xrange(n_hidden)] }

                sess.run(self._train, feed_dict=feed_dict)

                sum_cost += sess.run(self.loss, feed_dict={self.y: val_data,
                                                            self.user_index: u})
            
            avg_cost = sum_cost / self.n_users
            model_improved = True if old_avg_cost == None or old_avg_cost > avg_cost else False
            print 'Iteration: {}                    Average Cost per User: {}'.format(i, cost)
            iteration += 1
        sess.close()
   
    def recommend(self, u, Y, N=5, test_set=False):
        '''
           Method recommend outputs the top-N item recommendations for 
           user u
           
           Inputs:
               u (Int): id for user to be recommended
               Y (AbstractInputSource): user data class that encapsulates
                                        training, test, and full data
               N (Int): number of items to recommend user
               test_set (Boolean): necessary flag for checking the accuracy
                                   of the training set to the test set
           Output:
               List: Top N list of items recommend to user u  
        '''
                
        if test_set:
            if not isinstance(Y, AbstractInputClass):
                raise ValueError
            recommendation_candidates = Y.get_neg_indices()
            if self.sparse:
                train_data = self.sess.run(tf.sparse_tensor_to_dense(Y.get_train_data()))
            else:
                train_data = Y.get_train_data()
            
            feed_dict = {self.y: train_data}
        else:
            recommendation_candidates = np.where(Y == 0) 
            feed_dict = {self.y: Y}
        y_hat = self.sess.run(self.y_hat(u), feed_dict=feed_dict) 
        top_recommended_items = sorted(recommendation_candidates, key=lambda i: y_hat[i], reverse=True)
        return top_items_sorted[:N] if len(top_items_sorted) > N else top_items_sorted

    def _save_weights(self, with_error=False):
        output_dict = {}
        if with_error:
            # Mark the file as distinct  
            pass
        for weight in self.weight.items():
            weight = self.sess.run(weight)         
        
    def _load_weights(self):
        pass
    def _count_num_accurate_classification(self, test_set_items, recommendations):
        ''' 
            | Intersection(C-adopted, C-NRecommended) |
            Inputs:
                test_set_items (List): Actual items that user liked
                recommendation (List): Items the system recommended to the user
            Output:
                Int: Length of intersection between system recommended items and 
                     items the user actually liked 
        '''
        num_accurate_predictions = 0
        for item in recommendations:
            if item in test_set_items:
                num_accurate_predictions += 1
        return num_accurate_predictions

    def precision(self, actual, predicted, N=5):
        ''' 
            Calculates precision defined by: 
                | Intersection(C-adopted, C-NRecommended) | / N

            Inputs:
                actual (List): Actual items that were liked
                predicted (List): Prediction of actual items
            Output:
                Float: Number b/w 0 and 1 corresponding to the precision 
                       of the recommendation list
        '''
    
        if len(predicted) > N:
            predicted = predict[:N]

        num_accurate_predictions = self._count_num_accurate_classification(actual, predicted) 
        return float(num_accurate_predictions) / N

    def recall(self, actual, predicted, N=5):
        ''' 
            Calculates recall defined by: 
                | Intersection(C-adopted, C-NRecommended) | / | C-adopted | 

            Inputs:
                actual (List): Actual items that were liked
                predicted (List): Prediction of actual items
                N (Int): number of items to be recommended
            Output:
                Float: Number b/w 0 to 1 corresponding to the recall 
                       of the recommendation list
        '''
           
        num_accurate_predictions = self._count_num_accurate_classification(test_set_positive_items, recommendations)
        return float(num_accurate_predictions) / len(test_set_positive_items)

    def average_precision(self, u, Y, N):
        '''
            Average Precision at N is defined as:
                sum(Precision@k * Rel(K), k, 1, N) / min(N, |C-adopted|)
            Inputs: 
                u (Int): Represents user u
                Y (AbstractInputSource): user data class that encapsulates
                                         training, test, and full data
                N (Int): number of items to be recommended
            Output:
                Float; Number b/w 0 to 1 corresponding to the Average precision
                       of the recommendation list

        '''
        min_u_adopted = min(N, len(test_set_positive_items)) 
        actual = Y.get_test_data()
        predicted = self.recommend(u, Y, N=N, test_set=True)
        return (sum([self.precision(actual, predicted, N=k) if predicted[k] in actual else 0.0 for k in xrange(1, N+1)]) / 
                float(min_N_adopted))

    def mean_average_precision(self, N):
        '''
            Average of AP for all users

            Inputs:
                N (Int): Number of items to be recommended 
            Output:
                Float: Number b/w 0 to 1 corresponding to the Mean Average Precision@N
        '''
        return (sum([self.average_precision(u, user, N) for u, user in enumerate(self.input_data.get_data())]) / 
               float(len(self.users))) 

    def _create_variables(self, n_hidden=50):
        '''

            Initialize Tensorflow Variables
            Inputs:
                
            Outputs:
                
        '''
        # TODO-Look into slicing Big Variables and see if they are updatable.. Check performance        
        print 'Creating Tensorflow Variables...'
        self.user_index = tf.placeholder(tf.int32, shape=[])
        all_weights = dict()
        all_weights['W'] = tf.Variable(utils.xavier_init(self.n_items, n_hidden))
        all_weights['W_prime'] = tf.Variable(utils.xavier_init(self.n_items, n_hidden)) 
        all_weights['b'] = tf.Variable(tf.zeros([1, n_hidden], dtype=tf.float32))   
        all_weights['b_prime'] = tf.Variable(tf.zeros([1, self.n_items], dtype=tf.float32)) 
        all_weights['V'] = tf.Variable(tf.zeros([self.n_users, n_hidden], dtype=tf.float32))  
        all_weights['V_u'] = tf.nn.embedding_lookup(all_weights['V'], self.user_index)
        print 'Done.'
        return all_weights
