# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T
import theano
import TensorAlgebra as ta
import scipy.io as sio


class TensorInputLayer(object):
    def __init__(self, n_in, n_hid, n_out, W_in=None, W_out=None,
                 n_epochs=100, learning_rate=0.01, batch_size=None, 
                 l1=0.0, l2=0.0):
        
        rng = np.random.RandomState(1234)
        
        if W_in is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_hid)),
                                              high=np.sqrt(6. / (n_in + n_hid)),
                                              size=(n_in, n_hid)
                                              ), dtype=theano.config.floatX )
            self.W_in = theano.shared(value=W_values, name='W_in', borrow=True )
        else:
            self.W_in = theano.shared(value=W_in)
            
        if W_out is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_hid + n_out)),
                                              high=np.sqrt(6. / (n_hid + n_out)),
                                              size=(n_hid, n_out)
                                              ), dtype=theano.config.floatX )
            self.W_out = theano.shared(value=W_values,
                                       name='W_out', borrow=True )
        else:
            self.W_out = theano.shared(value=W_out)
            
        
        self.p_y_given_x = None
        self.y_pred = None
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        self.num_of_classes = n_out
        
    def negative_log_likelihood(self, y):
        l1_penalty = self.l1 * ( T.abs_(self.W_in).mean() + T.abs_(self.W_out).mean() )
        l2_penalty = self.l2 * ( (self.W_in**2).mean() + (self.W_out**2).mean() ) 
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + l1_penalty + l2_penalty

    
    def error(self, y):
        return -T.mean(T.neq(self.y_pred, y))
        
        
    def fit(self, input, labels):
        train_set_x = theano.shared(value = input)
        train_set_y = theano.shared(value = labels)
        
        if self.batch_size == None:
            self.batch_size = input.shape[0]
    
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        
        index = T.lscalar()
        x = T.tensor3('x')
        y = T.ivector('y')
        
        w_mul_x = self.W_in * x #T.dot(x, self.W_in)
        hidden_input = T.sum(w_mul_x, axis=1)
        hidden_output = T.tanh(hidden_input)
        out_input = T.dot(hidden_output, self.W_out)
        out_output = T.nnet.softmax(out_input)
        self.p_y_given_x = out_output
        
        cost = self.negative_log_likelihood(y)
        
        gW_in = T.grad(cost, wrt=self.W_in)
        #gW_out = T.grad(cost, wrt=self.W_out)
        
        updates = [(self.W_in, self.W_in - self.learning_rate * gW_in)]
        
        
        train_model = theano.function(inputs = [index],
                                      outputs = cost,
                                      updates = updates,
                                      givens = {x: train_set_x[index * self.batch_size:(index + 1)* self.batch_size],
                                                y: train_set_y[index * self.batch_size:(index + 1)* self.batch_size]})
    
        cost = np.inf
        epoch = 0
        while self.n_epochs >= epoch:
            epoch = epoch + 1
            if epoch % self.n_epochs == 0:
                    print cost
                    
            
            for minibatch_index in range(n_train_batches):
                cost = train_model(minibatch_index)
                
        return self.W_in.eval()        
        
        
    def predict(self, input):
        x = T.matrix('x')
    
        hidden_input = T.dot(x, self.W_in)
        hidden_output = T.tanh(hidden_input)
        out_input = T.dot(hidden_output, self.W_out)
        out_output = T.nnet.softmax(out_input)
        self.p_y_given_x = out_output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        test_model = theano.function(inputs=[x], outputs=self.y_pred)
        
        return test_model(input)
                
        
class TensorOutputLayer(object):
    def __init__(self, n_in, n_hid, n_out, W_in=None, W_out=None, b=None, u=None,
                 n_epochs=100, learning_rate=0.01, batch_size=None, 
                 l1=0.0, l2=0.0):
        
        rng = np.random.RandomState(1234)
        
        if W_in is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_hid)),
                                              high=np.sqrt(6. / (n_in + n_hid)),
                                              size=(n_in, n_hid)
                                              ), dtype=theano.config.floatX )
            self.W_in = theano.shared(value=W_values, name='W_in', borrow=True )
        else:
            self.W_in = theano.shared(value=W_in)
            
        if W_out is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_hid + n_out)),
                                              high=np.sqrt(6. / (n_hid + n_out)),
                                              size=(n_hid, n_out)
                                              ), dtype=theano.config.floatX )
            self.W_out = theano.shared(value=W_values,
                                       name='W_out', borrow=True )
        else:
            self.W_out = theano.shared(value=W_out)
            
        
        if b is None:
            self.b = theano.shared(value=np.ones((n_out,)))
        else:
            self.b = theano.shared(value=b)
            
        if u is None:
            self.u = theano.shared(value=np.ones((n_hid,)))
        else:
            self.u = theano.shared(value=u)
            
        
        self.p_y_given_x = None
        self.y_pred = None
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        self.num_of_classes = n_out
                
        
    def negative_log_likelihood(self, y):
        l1_penalty = self.l1 * ( T.abs_(self.W_in).mean() + T.abs_(self.W_out).mean() )
        l2_penalty = self.l2 * ( (self.W_in**2).mean() + (self.W_out**2).mean() ) 
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + l1_penalty + l2_penalty

    
    def error(self, y):
        return -T.mean(T.neq(self.y_pred, y))
        
        
    def fit(self, input, labels):
        train_set_x = theano.shared(value = input)
        train_set_y = theano.shared(value = labels)
        
        if self.batch_size == None:
            self.batch_size = input.shape[0]
    
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')
        
        hidden_input = T.dot(x, self.W_in) + self.u
        hidden_output = T.tanh(hidden_input)
        out_input = T.dot(hidden_output, self.W_out)
        out_output = T.nnet.softmax(out_input + self.b)
        self.p_y_given_x = out_output
        
        cost = self.negative_log_likelihood(y)
        
        gW_out = T.grad(cost, wrt=self.W_out)
        gb = T.grad(cost, wrt=self.b)
        gu = T.grad(cost, wrt=self.u)
        
        updates = [(self.W_out, self.W_out - self.learning_rate * gW_out),
                   (self.b, self.b - self.learning_rate * gb),
                   (self.u, self.u - self.learning_rate * gu)]
        
        
        train_model = theano.function(inputs = [index],
                                      outputs = cost,
                                      updates = updates,
                                      givens = {x: train_set_x[index * self.batch_size:(index + 1)* self.batch_size],
                                                y: train_set_y[index * self.batch_size:(index + 1)* self.batch_size]})
    
        cost = np.inf
        epoch = 0
        while self.n_epochs >= epoch:
            epoch = epoch + 1
            if epoch % self.n_epochs == 0:
                    print cost
                    
            
            for minibatch_index in range(n_train_batches):
                cost = train_model(minibatch_index)
                
        return self.W_out.eval(), self.b.eval(), self.u.eval(), cost        
        
        
    def predict(self, input):
        x = T.matrix('x')
    
        hidden_input = T.dot(x, self.W_in)
        hidden_output = T.tanh(hidden_input)
        out_input = T.dot(hidden_output, self.W_out)
        out_output = T.nnet.softmax(out_input, self.b)
        self.p_y_given_x = out_output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        test_model = theano.function(inputs=[x], outputs=self.y_pred)
        
        return test_model(input)        
        
        
class TensorNonLinear(object):
    def __init__(self, input_shape, n_hid, n_out, learning_rate=0.1, iterations=10, l1=0.0, l2=0.0):
        self.f1 = input_shape[1]
        self.f2 = input_shape[2]
        self.f3 = input_shape[0]
        self.n_hid = n_hid
        self.n_out = n_out
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w_out = None
        self.b = None
        
        self.w1_best = None
        self.w2_best = None
        self.w3_best = None
        self.w_out_best = None
        self.b_best = None

        self.matr_mode_1 = None
        self.matr_mode_2 = None
        self.matr_mode_3 = None
        self.x_vec = None
        self.tensor_mlp_1 = None
        self.tensor_mlp_2 = None
        self.tensor_mlp_3 = None
        self.tensor_mlp_out = None
        self.predictions = None
        self.iterations = iterations
        self.l1 = l1
        self.l2 = l2
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.min_cost = np.inf
        
    
    def fit(self, x_train, y_train, x_test=None, y_test=None):
        self.matr_mode_1 = np.asarray([ta.tensor_3d_matricization(x, mode=1) for x in x_train])
        self.matr_mode_2 = np.asarray([ta.tensor_3d_matricization(x, mode=2) for x in x_train])
        self.matr_mode_3 = np.asarray([ta.tensor_3d_matricization(x, mode=3) for x in x_train])
    
        self.x_vec = np.asarray([ta.vectorize_tensor(x) for x in x_train])
        
        rng = np.random.RandomState(1234)
        min_error = np.inf
        epsilon = 0.025
        
        self.w1 = np.asarray(rng.uniform(low=-np.sqrt(6. / (self.f1 + self.n_hid)),
                                            high=np.sqrt(6. / (self.f1 + self.n_hid)),
                                            size=(self.f1, self.n_hid)
                                            ), dtype=theano.config.floatX )
        self.w2 = np.asarray(rng.uniform(low=-np.sqrt(6. / (self.f2 + self.n_hid)),
                                            high=np.sqrt(6. / (self.f2 + self.n_hid)),
                                            size=(self.f2, self.n_hid)
                                            ), dtype=theano.config.floatX )
        self.w3 = np.asarray(rng.uniform(low=-np.sqrt(6. / (self.f3 + self.n_hid)),
                                            high=np.sqrt(6. / (self.f3 + self.n_hid)),
                                            size=(self.f3, self.n_hid)
                                            ), dtype=theano.config.floatX )
        self.w_out = np.asarray(rng.uniform(low=-np.sqrt(6. / (self.n_hid + self.n_out)),
                                            high=np.sqrt(6. / (self.n_hid + self.n_out)),
                                            size=(self.n_hid, self.n_out)
                                            ), dtype=theano.config.floatX )  
        self.b = np.ones((self.n_out,))
        self.u = np.ones((self.n_hid,))
        
        patience = 30
        for i in range(self.iterations):
            print('Iteration {}').format(i)
            
            self.tensor_mlp_1 = TensorInputLayer(self.f1, self.n_hid, self.n_out, 
                                                 W_in=self.w1, W_out=self.w_out,
                                                 n_epochs=300, learning_rate=self.learning_rate, 
                                                 batch_size=None, l1=0.0, l2=0.0)
            krp = ta.khatri_rao_product(self.w3,self.w2)
            x_temp = np.dot(self.matr_mode_1, krp)
            self.w1 = self.tensor_mlp_1.fit(x_temp, y_train)
           
            
            self.tensor_mlp_2 = TensorInputLayer(self.f2, self.n_hid, self.n_out, 
                                                 W_in=self.w2, W_out=self.w_out,
                                                 n_epochs=300, learning_rate=self.learning_rate, 
                                                 batch_size=None, l1=0.0, l2=0.0)
            krp = ta.khatri_rao_product(self.w3,self.w1)
            x_temp = np.dot(self.matr_mode_2, krp)
            self.w2 = self.tensor_mlp_2.fit(x_temp, y_train)
            
            
            self.tensor_mlp_3 = TensorInputLayer(self.f3, self.n_hid, self.n_out, 
                                                 W_in=self.w3, W_out=self.w_out,
                                                 n_epochs=300, learning_rate=self.learning_rate, 
                                                 batch_size=None, l1=0.0, l2=0.0)
            krp = ta.khatri_rao_product(self.w2,self.w1)
            x_temp = np.dot(self.matr_mode_3, krp)
            self.w3 = self.tensor_mlp_3.fit(x_temp, y_train)
            
            
            krp = ta.khatri_rao_product(self.w3,self.w2)
            w_in = ta.khatri_rao_product(krp,self.w1) 
            self.tensor_mlp_4 = TensorOutputLayer(self.x_vec.shape[1], self.n_hid, self.n_out, 
                                                 W_in=w_in, W_out=self.w_out, b=self.b, u=self.u,
                                                 n_epochs=300, learning_rate=self.learning_rate, 
                                                 batch_size=None, l1=0.0, l2=0.0)
            x_temp = self.x_vec
            self.w_out, self.b, self.u, cost = self.tensor_mlp_4.fit(x_temp, y_train)
            
            if cost < min_error:
                min_error = cost
                patience = 40
                self.w1_best = self.w1
                self.w2_best = self.w2
                self.w3_best = self.w3
                self.w_out_best = self.w_out
                self.b_best = self.b
                
            else:
                patience = patience - 1
                
            if patience == 0:
                self.w1 = self.w1_best
                self.w2 = self.w2_best
                self.w3 = self.w3_best
                self.w_out = self.w_out_best
                self.b = self.b_best
                break
            
            y_pred = self.predict(x_test)
            errors_test = np.sum(np.not_equal(y_pred, y_test)) / np.float(x_test.shape[0])
            
            if errors_test < self.min_cost:
                self.min_cost = errors_test
            
            y_pred = self.predict(x_train)
            errors_train = np.sum(np.not_equal(y_pred, y_train)) / np.float(x_train.shape[0])
            
            print('..... Training error {} - Testing error {}').format(errors_train, errors_test)
            print('..... Minimum testing error {}').format(self.min_cost)



    def predict(self, x_test):
        krp = ta.khatri_rao_product(self.w3,self.w2)
        w_in = ta.khatri_rao_product(krp,self.w1)
        x_vec = np.asarray([ta.vectorize_tensor(x) for x in x_test])
        
        W_in = theano.shared(value=w_in)
        W_out = theano.shared(value=self.w_out)
        x = T.matrix('x')
             
        hidden_input = T.dot(x, W_in) + self.u
        hidden_output = T.tanh(hidden_input)
        out_input = T.dot(hidden_output, W_out)
        out_output = T.nnet.softmax(out_input + self.b)
        self.p_y_given_x = out_output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        test_model = theano.function(inputs=[x], outputs=self.y_pred)
        
        return test_model(x_vec)       
        
        
        
def preprocess_data(r_mat, r_dict, gt_mat, gt_dict):
    datasets = sio.loadmat(r_mat)
    hypercube = datasets[r_dict]
    
    ##### Standardize ####
    sz_0 = hypercube.shape[0]
    sz_1 = hypercube.shape[1]
    sz_2 = hypercube.shape[2]
    hypercube = np.reshape(hypercube, (-1, sz_2))
    h_mean = np.mean(hypercube, axis=0)
    hypercube = hypercube - h_mean
    hypercube = np.reshape(hypercube, (sz_0, sz_1, sz_2))

    datasets = sio.loadmat(gt_mat)
    ground_truth = datasets[gt_dict]

    window_sz = 5
    window_pad = 2
    gt_nonzero = np.count_nonzero(ground_truth)
    dataset_matrix_size = (gt_nonzero, window_sz, window_sz, hypercube.shape[2])
    dataset_matrix = np.zeros(dataset_matrix_size)
    label_vector = np.zeros((dataset_matrix.shape[0],))

    data_image = []
    data_index = 0
    for r in range(hypercube.shape[0]):
        if r < window_pad or r > hypercube.shape[0] - window_pad-1:
            continue
        for c in range(hypercube.shape[1]):
            if c < window_pad or c > hypercube.shape[1] - window_pad-1:
                continue
            if ground_truth[r,c] == 0:
                patch = hypercube[r-window_pad:r+window_pad+1, c-window_pad:c+window_pad+1]
                data_image.append(patch)
                continue
        
            patch = hypercube[r-window_pad:r+window_pad+1, c-window_pad:c+window_pad+1]
            data_image.append(patch)
            dataset_matrix[data_index,:,:,:] = patch
            label_vector[data_index] = ground_truth[r,c]        
        
            data_index = data_index + 1

    rand_perm = np.random.permutation(label_vector.shape[0])
    dataset_matrix = dataset_matrix[rand_perm,:,:,:]
    label_vector = label_vector[rand_perm]
    
    label_vector = label_vector - 1.0
    
    data_image = np.asarray(data_image)
    data_image = data_image / np.max(dataset_matrix)
    return dataset_matrix/np.max(dataset_matrix), label_vector, data_image
        
        
def load_data_multi200(dataset_matrix_r, label_vector_r, n_components, n_classes, samples):
    
    def shared_dataset(data_x, data_y, borrow=True):
        x_ar = np.reshape(data_x, (-1, 5*5*n_components) )
        shared_x = theano.shared(np.asarray(x_ar, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')
    
    s_sum = 0
    for i in range(n_classes):
        idx = np.where(label_vector_r == i)   
        if idx[0].shape[0] < samples:
            s_samples = idx[0].shape[0]*4 / 5
        else:
            s_samples = samples
        s_sum = s_sum + s_samples
    
    test_set_200 = np.zeros((dataset_matrix_r.shape[0] - s_sum, dataset_matrix_r.shape[1], dataset_matrix_r.shape[2], dataset_matrix_r.shape[3]))
    l_test_set_200 = np.zeros((dataset_matrix_r.shape[0] - s_sum))
    train_set_200 = np.zeros((s_sum, dataset_matrix_r.shape[1], dataset_matrix_r.shape[2], dataset_matrix_r.shape[3]))    
    l_train_set_200 = np.zeros((s_sum))
    
    count_start = 0
    count_train = 0
    for i in range(n_classes):
        idx = np.where(label_vector_r == i)
        rand_perm = np.random.permutation(idx[0].shape[0])
        class_i = dataset_matrix_r[idx]
        class_i = class_i[rand_perm]
        
        if idx[0].shape[0] < samples:
            s_samples = idx[0].shape[0]*4 / 5
        else:
            s_samples = samples
        
        count_end = count_start + idx[0].shape[0] - s_samples        
        
        train_set_200[count_train:count_train+s_samples,:] = class_i[0:s_samples]
        l_train_set_200[count_train:count_train+s_samples] = i
        count_train = count_train+s_samples
        
        test_set_200[count_start:count_end,:] = class_i[s_samples:]
        l_test_set_200[count_start:count_end] = i
        
        count_start = count_end
        
    rand_perm = np.random.permutation(l_train_set_200.shape[0])
    l_train_set_200 = l_train_set_200[rand_perm]
    train_set_200 = train_set_200[rand_perm]
    
    rand_perm = np.random.permutation(l_test_set_200.shape[0])
    l_test_set_200 = l_test_set_200[rand_perm]
    test_set_200 = test_set_200[rand_perm]
        
    test_set_x, test_set_y = shared_dataset(train_set_200, l_train_set_200)
    valid_set_x, valid_set_y = shared_dataset(train_set_200, l_train_set_200)
    train_set_x, train_set_y = shared_dataset(train_set_200, l_train_set_200)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval 
        


if __name__=='__main__':   
         
    dataset_mat = 'PaviaU.mat'
    dataset_dict = 'paviaU'
    labels_mat = 'PaviaU_gt.mat'
    labels_dict = 'paviaU_gt' 
    
    patch_size = 5
    bands = 103
    classes = 9
    hidden_neurons = 100
    
    data, labels, data_image = preprocess_data(dataset_mat, dataset_dict, labels_mat, labels_dict)  
    
    rval = load_data_multi200(data, labels, bands, classes, 50) #bands - classes - samples per class
        
    X_train = np.reshape(rval[0][0].eval(), (-1,patch_size,patch_size,bands))
    y_train = rval[0][1].eval()
    
    clf = TensorNonLinear([patch_size,patch_size,bands], hidden_neurons, classes, learning_rate=0.05, iterations=500, l1=0.0, l2=0.0)
    clf.fit(X_train, y_train, data, labels)
    
    y_pred = clf.predict(data)
    errors = np.sum(np.not_equal(y_pred, labels))
    
    

    
   
   
   
   
