import matplotlib.pyplot as plt
import numpy as np
import re
import os
import datetime as dt

k = 0.25

path = r"E:\Pattern Recognition\price_pridiction\futuresData\\"
for f in os.listdir(path):
    if f.find("0-2017") != -1:
        c = ['A1', 'A3']
    elif f.find("1-2017") != -1:
        c = ['B2', 'B3']
    else:
        continue
    fr = open(path + f, 'r')
    fname = re.split('[-.]', f)
    fw = [None, None]
    fw[0] = open(path + fname[1] + fname[2] + '-' + c[0] + '.log1', 'w')
    fw[1] = open(path + fname[1] + fname[2] + '-' + c[1] + '.log1', 'w')
    line = fr.readline()
    if re.search('2017/.*2017-', line):
        line = re.sub('2017/.*2017-', '2017-', line)
    lines = line.split(',')
    while True:
        t = lines[0].split()[1]
        t = re.split('[:.]', t)
        if t[0] not in ['08', '20']:
            break
        line = fr.readline()
        if re.search('2017/.*2017-', line):
            line = re.sub('2017/.*2017-', '2017-', line)
        lines = line.split(',')
    v13 = [0, 0]
    t13 = [0, 0]
    tovr13 = [0, 0]
    p13 = [0, 0]
    while True:
        t = lines[0].split()[1]
        t = re.split('[:.]', t)
        tovr = int(lines[4].split('=')[1]) // 1000
        v = int(lines[3].split('=')[1])
        if lines[9].split('=')[1] == c[0]:
            tovr13[0], v13[0], t13[0] = tovr, v, float(t[2]) + 0.501
        elif lines[9].split('=')[1] == c[1]:
            tovr13[1], v13[1], t13[1] = tovr, v, float(t[2]) + 0.501
        if tovr13[0] != 0 and tovr13[1] != 0:
            break
        line = fr.readline()
        if re.search('2017/.*2017-', line):
            line = re.sub('2017/.*2017-', '2017-', line)
        lines = line.split(',')
    for line in fr:
        if re.search('2017/.*2017-', line):
            line = re.sub('2017/.*2017-', '2017-', line)
        lines = line.split(',')
        t = lines[0].split()[1]
        t = re.split('[:.]', t)
        tovr = int(lines[4].split('=')[1]) // 1000
        v = int(lines[3].split('=')[1])
        if lines[9].split('=')[1] == c[0]:
            idx = 0
        elif lines[9].split('=')[1] == c[1]:
            idx = 1
        else:
            continue
        if float(t[2]) > t13[idx]:
            #write last line with v=0
            if p13[idx] > 0.1:
                fw[idx].write("%.3f\t%d\n" % (p13[idx], 0))
            t13[idx] = float(t[2]) + 0.001
            pass
        #write new p,v
        bid = int(lines[5].split('=')[1]) // 1000
        ask = int(lines[7].split('=')[1]) // 1000
        if ask == 0 or bid == 0:
            p = int(lines[0].split('=')[1]) // 1000
        elif v == v13[idx]:
            p = k * int(lines[0].split('=')[1]) // 1000 + (1 - k) * (bid + ask) / 2
        else:
            p = k * (tovr - tovr13[idx]) / (v - v13[idx]) + (1 - k) * (bid + ask) / 2
        fw[idx].write("%.3f\t%d\n" % (p, v - v13[idx]))
        t13[idx] = (t13[idx] + 0.5) % 60
        v13[idx] = v
        p13[idx] = p
        tovr13[idx] = tovr
    fr.close()
    fw[0].close()
    fw[1].close()
