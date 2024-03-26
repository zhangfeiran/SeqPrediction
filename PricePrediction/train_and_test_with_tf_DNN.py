import matplotlib.pyplot as plt
import numpy as np
import re
import os
import random
import tensorflow as tf
import sklearn.metrics as sm
import pickle

iid = 3
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

feature_columns = [tf.contrib.layers.real_valued_column("x", dimension=dim)]
classifier=tf.estimator.DNNClassifier(
    hidden_units=[128,128],
    feature_columns=feature_columns,
    n_classes=3,
    model_dir='tmp/',
    optimizer=tf.train.AdamOptimizer(beta1=0.8),
)

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
            # v.append(float(lines[1]))
            y = lines[2]
            if len(p) > dim:
                d = [i - p[-1] for i in p[-dim - 1:-1]]
                # d2 = [i - v[-1] for i in v[-dim - 1:-1]]
                if lines[2] == '-1':
                    num[0] += 1
                    data[0].append((d, y))
                elif lines[2] == '1':
                    num[2] += 1
                    data[2].append((d, y))
                elif random.random() < 0.05:
                    num[1] += 1
                    data[1].append((d, y))
                    # if min(d) > -200 and max(d) < 200:
                    #     if random.random() < 0.001:
                    #         plt.plot(d, 'b',alpha=0.5)
    print(num)
    size = [int(min(num) * 0.6), int(min(num))]
    data = [random.sample(i, size[1]) for i in data]
    with open(instrument[iid]+'size.dat','wb') as fw:
        pickle.dump(size,fw)
        # np.savetxt('size.txt',size,fmt='%d')
    with open(instrument[iid]+'data.dat','wb') as fw:
        pickle.dump(data,fw)
    exit()
else:
    with open(instrument[iid]+'size.dat','rb') as fr:
        size=pickle.load(fr)
    with open(instrument[iid]+'data.dat','rb') as fr:
        data=pickle.load(fr)

trainingData = [data[i][j][0] for i in range(3) for j in range(0, size[0])]
trainingLabels = [int(data[i][j][1]+1) for i in range(3) for j in range(0, size[0])]
print(len(trainingData))
testData = [data[i][j][0] for i in range(3) for j in range(size[0], size[1])]
testLabels = [int(data[i][j][1]+1) for i in range(3) for j in range(size[0], size[1])]


def train_input_fn():
    x={'x':tf.constant(trainingData)}
    y=tf.constant(trainingLabels)
    return x,y
def test_input_fn():
    x={'x':tf.constant(testData)}
    y=tf.constant(testLabels)
    return x,y

print('Training...')
classifier.train(train_input_fn,steps=2000)
print('Testing...')
predictLabels = classifier.predict(input_fn=test_input_fn)
# predictLabels = [i for i in predictLabels]
predictLabels = [int(predictLabels.__next__()['classes'][0]) for i in range(len(testLabels))]
# clf.fit(trainingData, trainingLabels)

# predictLabels = clf.predict(testData)
result = sm.classification_report(testLabels, predictLabels)
print(result)
print(sm.confusion_matrix(testLabels, predictLabels))

# predictLabels = clf.predict(trainingData)
# result = sm.classification_report(trainingLabels, predictLabels)
# print(sm.confusion_matrix(trainingLabels, predictLabels))
# print(result)