import tensorflow as tf
from tensorflow.keras.backend import binary_crossentropy
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.nn import bidirectional_dynamic_rnn, embedding_lookup, dropout
from tensorflow.contrib.rnn import LSTMCell
from tqdm import tqdm_notebook
from utils import *

class MaLSTM:
    def __init__(self, emb_matrix, emb_trainable=False):
        tf.reset_default_graph()
                
        # input
        self.input_sentA = tf.placeholder(tf.int32, shape=[None, None])  # (batch_size, time_step)
        self.input_sentB = tf.placeholder(tf.int32, shape=[None, None])  # (batch_size, time_step) 
        self.input_seq_lenA = tf.placeholder(tf.int32, shape=[None, ])  # (batch_size, )
        self.input_seq_lenB = tf.placeholder(tf.int32, shape=[None, ])  # (batch_size, )
        
        # output
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, ])  # (batch_size, )

        # embedding matrix
        self.embedding_matrix = tf.get_variable(shape=emb_matrix.shape, 
                                    initializer=tf.constant_initializer(emb_matrix, dtype=tf.float32),
                                    dtype=tf.float32,
                                    trainable=emb_trainable,
                                    name='embeddings_matrix')
        
        
    def embedding_layer(self, sequence):
        return embedding_lookup(self.embedding_matrix, sequence)  # (batch_size, time_step, emb_dim)
    
    
    def bilstm(self, sequence, sequence_length, lstm_unit, reuse=None):
        with tf.variable_scope('BiLSTM', reuse=reuse, dtype=tf.float32):
            cell_fw = LSTMCell(num_units=lstm_unit, reuse=tf.get_variable_scope().reuse)
            cell_bw = LSTMCell(num_units=lstm_unit, reuse=tf.get_variable_scope().reuse)
            
        ((output_fw, output_bw), _) = bidirectional_dynamic_rnn(cell_fw, cell_bw, sequence, dtype=tf.float32, sequence_length=sequence_length)
        
        return tf.concat([output_fw, output_bw], axis=2)  # (batch_size, num_step, lstm_unit * 2)
    
    
    def lstm(self, sequence, sequence_length, lstm_unit, reuse=None):
        with tf.variable_scope('LSTM', reuse=reuse, dtype=tf.float32):
            cell = tf.contrib.rnn.LSTMCell(num_units=lstm_unit, reuse=tf.get_variable_scope().reuse)

        _, state = tf.nn.dynamic_rnn(cell, sequence, dtype=tf.float32, sequence_length=sequence_length)
        return state[1]  # (batch_size, lstm_unit)
    
    
    def dense(self, input_mat, unit, scope, activation='sigmoid'):
        input_dim = (input_mat.get_shape().as_list()[1])
        
        w_shape, b_shape = (input_dim, unit), (unit, )
        with tf.variable_scope(scope):
#             weight = tf.get_variable(initializer=tf.random_normal(w_shape), name='weight')
#             bias = tf.get_variable(initializer=tf.random_normal(b_shape), name='bias')
            weight = tf.get_variable(initializer=xavier_initializer(), shape=w_shape, dtype=tf.float32, name='output_weight')
            bias = tf.get_variable(initializer=xavier_initializer(), shape=b_shape, dtype=tf.float32, name='output_bias')
        
        out = tf.matmul(input_mat, weight) + bias
        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)
        
        return out  # (batch_size, unit)
    
    
    def loss_function(self, output):
        cross_entropy = binary_crossentropy(target=self.target, output=output)  # (batch_size, )
        return tf.reduce_mean(cross_entropy)  # ()
    
    
    def build(self, lstm_unit=256, hidden_unit=16, output_unit=1, learning_rate=0.001, encoder='lstm'):
        word_embA = self.embedding_layer(self.input_sentA)  # (batch_size, num_step, emb_dim)
        word_embB = self.embedding_layer(self.input_sentB)  # (batch_size, num_step, emb_dim)
        
        if encoder == 'lstm':
            repA = self.lstm(word_embA, self.input_seq_lenA, lstm_unit, None)  # (batch_size, lstm_unit)
            repB = self.lstm(word_embB, self.input_seq_lenB, lstm_unit, True)  # (batch_size, lstm_unit)
            input_dim = lstm_unit * 2
        elif encoder == 'bilstm':
            repA = self.bilstm(word_embA, self.input_seq_lenA, lstm_unit, None)  # (batch_size, num_step, lstm_unit * 2)
            repB = self.bilstm(word_embA, self.input_seq_lenB, lstm_unit, True)  # (batch_size, num_step, lstm_unit * 2)
            repA = tf.reduce_sum(repA, axis=1)  # (batch_size, lstm_unit * 2)
            repB = tf.reduce_sum(repB, axis=1)  # (batch_size, lstm_unit * 2)
            input_dim = lstm_unit * 4
        
        rep = tf.concat([repA, repB], axis=1)  # lstm: (batch_size, lstm_unit * 2), bilstm: (batch_size, lstm_unit * 4)
        rep = dropout(rep, keep_prob=0.5)
        
        hidden = self.dense(rep, hidden_unit, 'hidden')  # (batch_size, hidden_unit)
        hidden = dropout(hidden, keep_prob=0.5)
        
        self.output = self.dense(hidden, output_unit, 'output')  # (batch_size, output_unit)
        self.output = tf.reshape(self.output, (-1, ))
        
        self.loss = self.loss_function(self.output)  # ()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        
    def fit(self, x_train, y_train, train_sequence_length, x_val, y_val, val_sequence_length, epoch_size, word_to_index, batch_size, model_name):
        def learn(X, Y, sequence_length, epoch, mode):
            tn = tqdm_notebook(total=len(X))
            for sentA, sentB, seq_lenA, seq_lenB, target in next_batch_with_pad(X, Y, sequence_length, word_to_index, batch_size):
                 
                feed_dict = {
                    self.input_sentA: sentA,
                    self.input_sentB: sentB, 
                    self.input_seq_lenA: seq_lenA,
                    self.input_seq_lenB: seq_lenB,
                    self.target: target
                }
                if mode == 'train':
                    fetches = [self.loss, self.output, self.optimizer]
                    loss, output, _ = self.sess.run(fetches, feed_dict)
                    tn.set_description('Epoch: {}/{}'.format(epoch, epoch_size))
                elif mode == 'validate':
                    fetches = [self.loss, self.output]
                    loss, output = self.sess.run(fetches, feed_dict)
                                
                tn.set_postfix(loss=loss, accuracy=accuracy(output, target), mode=mode)
                tn.update(n=batch_size)
                
        
        print('Train on {} samples, validate on {} samples'.format(len(x_train), len(x_val)))
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(1, epoch_size + 1):                
            # train
            learn(x_train, y_train, train_sequence_length, epoch, 'train')

            # validate
            learn(x_val, y_val, val_sequence_length, epoch, 'validate')    

        save_path = saver.save(self.sess, 'models/{}.ckpt'.format(model_name))
        print('Model was saved in {}'.format(save_path))
            
    
    def restore(self, model_path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, model_path)
            
    
    def predict(self, X, sequence_length, word_to_index):
        
        y_empty = np.empty(0)
        batch_size, i = 100, 0
        tn = tqdm_notebook(total=len(X))
        prediction = np.empty((len(X), ))
        for sentA, sentB, seq_lenA, seq_lenB, _ in next_batch_with_pad(X, y_empty, sequence_length, word_to_index, batch_size):
            fetches = [self.output]
            feed_dict = {
                self.input_sentA: sentA,
                self.input_sentB: sentB, 
                self.input_seq_lenA: seq_lenA,
                self.input_seq_lenB: seq_lenB,
            }
            output = self.sess.run(fetches, feed_dict)[0]
            prediction[i * batch_size: i * batch_size + len(output)] = np.round(output)
            
            tn.set_postfix(mode='predict')
            tn.update(n=batch_size)
            
            i += 1
        
        
        return prediction
        
        
