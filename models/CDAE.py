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
        #optimizer = tf.train.AdagradOptimizer(ada_learning_rate, ada_beta)
        optimizer = tf.train.GradientDescentOptimizer(ada_learning_rate)

        # Regularization Parameter
        reg_lambda = kwargs.get('reg_lambda') if kwargs.get('reg_lambda') else 1.0

        # Model Ops
        self.y = tf.placeholder(tf.float32, [None, self.n_items])
        self.z = lambda u: transfer_fn(tf.add_n( [tf.matmul(self.y, self.weight['W']), 
                                                  tf.reshape(self.weight['V'][u], [1, n_hidden]), 
                                                  tf.reshape(self.weight['b'], [1, n_hidden])]
                                                ))
        self.y_hat = lambda u: output_fn(tf.add(tf.matmul(self.z(u), tf.transpose(self.weight['W_prime'])),
                                                tf.reshape(self.weight['b_prime'], [1, self.n_items]))
                                        )   
        # Loss Function
        # TODO - Figure out how to include logistic loss efficiently
        self.loss = lambda u: tf.nn.l2_loss(tf.sub(self.y, self.y_hat(u)))

        # Gradient Ops
        #   - To my knowledge this when cost_gradient_apply gets called per user per item
        #     we are simulating online training. However, if we dont call apply_gradients
        #     we may be able to accumalate the gradients and apply them on a per user basis
        #     to simulate something like batch training.
        #   - TODO compare results of both online and batch methods
        
        self.dLoss_d = lambda u, var: optimizer.compute_gradients(self.loss(u), var_list=[var])[0][0]
        self.dCost_d = lambda u, var: tf.mul(1.0 / self.n_users, self.dLoss_d(u, var)) - tf.mul(reg_lambda, var)
        self.cost_gradient_apply = lambda u, var: optimizer.apply_gradients([[self.dCost_d(u, var), var]])
        self.init_op = tf.initialize_all_variables()
        print 'Done.'
 
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
        return aug_set

    def train(self, **kwargs):
        '''
            Method  
        '''
        MAX_ITERATION = kwargs.get('max_iteration') if kwargs.get('max_iteration') else 1000
        dropout_prob = kwargs.get('dropout_prob') if kwargs.get('dropout_prob') else 0.2
        n_hidden = kwargs.get('n_hidden') if kwargs.get('n_hidden') else 50
        dropout_prob = tf.constant(dropout_prob, dtype=tf.float32)
        iteration = 0

        self.weight = self._init_variables(n_hidden) 
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
                val_data = input_data[u].get_full_data()
                train_data = input_data[u].get_train_data()
                
                if self.sparse:
                    try:
                        val_data = sess.run(tf.sparse_tensor_to_dense(val_data))
                        train_data = sess.run(tf.sparse_tensor_to_dense(train_data))
                    except:
                        print 'user: {}'.format(u)
                y_tilde = sess.run(tf.nn.dropout(tf.constant(train_data, dtype=tf.float32, shape=[1, self.n_items]), dropout_prob))
                augmented_O_set = self._create_augmented_O_set(np.array(val_data))

                for item in augmented_O_set:
                    sess.run(self.cost_gradient_apply(u, self.weight['W_prime'][item]), feed_dict={self.y: y_tilde})
                    sess.run(self.cost_gradient_apply(u, self.weight['b_prime'][item]), feed_dict={self.y: y_tilde})
                for item in np.where(x_tilde == 1):
                    sess.run(self.cost_gradient_apply(u, self.weight['W'][item]), feed_dict={self.y: y_tilde})

                sess.run(self.cost_gradient_apply(u, self.weight['V'][u]), feed_dict={self.y: y_tilde})
                sess.run(self.cost_gradient_apply(u, self.weight['b']), feed_dict={self.y: y_tilde})
                sum_cost += sess.run(self.loss(u), feed_dict={self.y: val_data})
            
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

    def _init_variables(self, n_hidden=50):
        '''

            Initialize Tensorflow Variables
            Inputs:
                
            Outputs:
                
        '''
        # TODO-Look into slicing Big Variables and see if they are updatable.. Check performance        
        print 'Creating Tensorflow Variables...'
        W = utils.xavier_init(self.n_items, n_hidden)
        W_prime = utils.xavier_init(self.n_items, n_hidden)
        all_weights = dict()
        all_weights['W'] = [tf.Variable(W[i]) for i in xrange(self.n_items)]
        all_weights['W_prime'] = [tf.Variable(W_prime[i]) for i in xrange(self.n_items)]
        all_weights['b'] = tf.Variable(tf.zeros([1, n_hidden], dtype=tf.float32))   
        all_weights['b_prime'] = [tf.Variable(tf.constant(0, dtype=tf.float32), dtype=tf.float32) for _ in xrange(self.n_items)]
        all_weights['V'] = [tf.Variable(tf.zeros([n_hidden])) for _ in xrange(self.n_users)] 
        print 'Done.'
        return all_weights
