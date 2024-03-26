import matplotlib.pyplot as plt
import numpy as np
import re
import os
import random
import tensorflow as tf
import sklearn.metrics as sm
import pickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess = tf.Session()

lr = 1e-3
input_size = 1     
timestep_size = 100   
hidden_size = 128    
layer_num = 2        
class_num = 3       
cell_type = "lstm"   

X_input = tf.placeholder(tf.float32, [None, 100])
y_input = tf.placeholder(tf.float32, [None, class_num])

batch_size = tf.placeholder(tf.int32, [])  
keep_prob = tf.placeholder(tf.float32, [])

X = tf.reshape(X_input, [-1, 100, 1])

def lstm_cell(cell_type, num_nodes, keep_prob):
    assert(cell_type in ["lstm", "block_lstm"], "Wrong cell type.")
    if cell_type == "lstm":
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(num_nodes)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(cell_type, hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# h_state = state[-1][1]
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        (cell_output, state) = mlstm_cell(X[:, timestep, :],state)
        outputs.append(cell_output)
h_state = outputs[-1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
cross_entropy = -tf.reduce_mean(y_input * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


iid = 0
instrument = ['', '', '', '']
instrument[0] = 'A1'
instrument[1] = 'A3'
instrument[2] = 'B2'
instrument[3] = 'B3'
print('predicting ' + instrument[iid] + ' instrument')

dim = 100
data = [[], [], []]
np.random.seed(42)
random.seed(42)


save = False
save = True
if save:
    num = [0, 0, 0]
    path = r"E:\Pattern Recognition\price_pridiction\futuresData\\"
    for f in os.listdir(path):
        if f.find(instrument[iid] + ".log2") == -1:
            continue
        p, v = [], []
        for line in open(f, 'r'):
            lines = line.split()
            p.append(float(lines[0]))
            v.append(float(lines[1]))
            y = lines[2]
            if len(p) > dim:
                d = [i - p[-1] for i in p[-dim - 1:-1]]
                d2 = [i - v[-1] for i in v[-dim - 1:-1]]
                if lines[2] == '-1':
                    num[0] += 1
                    data[0].append((d,d2, y))
                elif lines[2] == '1':
                    num[2] += 1
                    data[2].append((d,d2, y))
                elif random.random() < 0.05:
                    num[1] += 1
                    data[1].append((d,d2, y))
                    # if min(d) > -200 and max(d) < 200:
                    #     if random.random() < 0.001:
                    #         plt.plot(d, 'b',alpha=0.5)
    print(num)
    size = [int(min(num) * 0.6), int(min(num))]
    data = [random.sample(i, size[1]) for i in data]
    with open(instrument[iid]+'size.dat','wb') as fw:
        pickle.dump(size,fw)
    with open(instrument[iid]+'data.dat','wb') as fw:
        pickle.dump(data,fw)
    # exit()
else:
    with open(instrument[iid]+'size.dat','rb') as fr:
        size=pickle.load(fr)
    with open(instrument[iid]+'data.dat','rb') as fr:
        data=pickle.load(fr)

trainingData = [data[i][j][0] for i in range(3) for j in range(0, size[0])]
trainingData=np.array(trainingData).reshape(size[0]*3,dim)
trainingData=tf.constant(trainingData)
trainingLabels = [[int((int(data[i][j][2])+1)==0),int((int(data[i][j][2])+1)==1),int((int(data[i][j][2])+1)==2)] for i in range(3) for j in range(0, size[0])]
trainingLabels=np.array(trainingLabels).reshape(size[0]*3,3)
trainingLabels=tf.constant(trainingLabels)
# print(len(trainingData))
testData = [data[i][j][0] for i in range(3) for j in range(size[0], size[1])]
testData=np.array(testData).reshape((size[1]-size[0])*3,dim)
testData=tf.constant(testData)
testLabels = [[int((int(data[i][j][2])+1)==0),int((int(data[i][j][2])+1)==1),int((int(data[i][j][2])+1)==2)] for i in range(3) for j in range(size[0], size[1])]
testLabels=np.array(testLabels).reshape((size[1]-size[0])*3,3)
testLabels=tf.constant(testLabels)

dataset=tf.data.Dataset.from_tensor_slices((trainingData,trainingLabels))
dataset=dataset.shuffle(50000).apply(tf.contrib.data.batch_and_drop_remainder(200)).repeat()
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

dataset2=tf.data.Dataset.from_tensor_slices((testData,testLabels))
dataset2=dataset2.shuffle(40000).apply(tf.contrib.data.batch_and_drop_remainder(1200)).repeat()
iterator2=dataset2.make_one_shot_iterator()
one_element2=iterator2.get_next()


sess.run(tf.global_variables_initializer())
print('start....................')
for i in range(500):
    _batch_size=200
    X_batch, y_batch = sess.run(one_element)
    cost, acc,  _ = sess.run([cross_entropy, accuracy, train_op], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 1.0, batch_size: _batch_size})
    # print(acc,_)
    if (i+1) % 100 == 0:
        print(i+1,cost,acc)

test_acc = 0.0
test_cost = 0.0
for i in range(20):
    X_batch, y_batch = sess.run(one_element2)
    _cost, _acc = sess.run([cross_entropy, accuracy], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 1.0, batch_size:1200 })
    test_acc += _acc
    test_cost += _cost
    print("test cost={:.6f}, acc={:.6f}".format(_cost, _acc))
print(test_acc/20)









