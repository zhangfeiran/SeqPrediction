import matplotlib.pyplot as plt
import numpy as np
import re
import os
import datetime as dt

a, b = 10,40
theta = 0.15 / 100

path = r"E:\Pattern Recognition\price_pridiction\futuresData\\"
for f in os.listdir(path):
    # if f.find("A1.log1") == -1:
    #     if f.find("A3.log1") == -1:
    #         if f.find("B2.log1") == -1:
    #             if f.find("B3.log1") == -1:
    #                 continue
    if f.find("B3.log1") == -1:
        continue
    p, v = [], []
    for line in open(f, 'r'):
        lines = line.split('\t')
        p.append(float(lines[0]))
        v.append(float(lines[1]))
    fw = open(path + 'labeled-' + f.split('.')[0] + '.log2', 'w')
    for i in range(1, len(p) - b - 1):
        d = [abs(p[i + j] - p[i]) for j in range(a, b)]
        ts = np.argmax(d)
        d = p[i + a + ts] - p[i]
        if d > theta * p[i]:
            label = 1
        elif d < -theta * p[i]:
            label = -1
        else:
            label = 0
        fw.write("%.3f\t%d\t%d\t%.3f\n" % (p[i], v[i], label, d * 100 / p[i]))
    fw.close()