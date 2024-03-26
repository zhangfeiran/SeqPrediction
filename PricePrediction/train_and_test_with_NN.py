import matplotlib.pyplot as plt
import numpy as np
import re
import os
import random
import pickle
import sklearn.metrics as sm
import sklearn.neural_network as nn
import sklearn.model_selection as sms
import sklearn.ensemble as se
import sklearn.neighbors as sn

clf = nn.MLPClassifier(
    hidden_layer_sizes=(256,128,64),
    alpha=0.0001,
    batch_size=200,
    learning_rate_init=0.001,
    max_iter=300,
    tol=1e-3,
    beta_1=0.95,
    beta_2=0.999,
)
# clf=sn.KNeighborsClassifier(3)
print(clf)
# clf=se.BaggingClassifier(clf)


iid = 3
instrument=['','','','']
instrument[0] = 'A1'
instrument[1] = 'A3'
instrument[2] = 'B2'
instrument[3] = 'B3'
print('predicting '+instrument[iid]+' instrument')

dim = 300
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
trainingLabels = [data[i][j][2] for i in range(3) for j in range(0, size[0])]
print(len(trainingData))
testData = [data[i][j][0] for i in range(3) for j in range(size[0], size[1])]
testLabels = [data[i][j][2] for i in range(3) for j in range(size[0], size[1])]

print('Training...')
clf.fit(trainingData, trainingLabels)

print('Testing...')
predictLabels=clf.predict(testData)
result = sm.classification_report(testLabels, predictLabels)
print(result)
print(sm.confusion_matrix(testLabels, predictLabels))

# predictLabels=clf.predict(trainingData)
# result = sm.classification_report(trainingLabels, predictLabels)
# print(result)
# print(sm.confusion_matrix(trainingLabels, predictLabels))
