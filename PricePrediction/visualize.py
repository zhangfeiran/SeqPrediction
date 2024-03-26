import matplotlib.pyplot as plt
import numpy as np

p,v=[],[]
for line in open('20170703day-B2.log1','r'):
    lines=line.split('\t')
    p.append(float(lines[0]))
    v.append(int(lines[1]))

pp=[]
for line in open('labeled-20170703day-B2.log2','r'):
    lines=line.split('\t')
    pp.append(float(lines[0]))

# plt.plot(list(range(0,len(p))),p)
# plt.plot(list(range(0,len(pp))),pp)
# plt.show()
# exit()        

n=len(p)//500-1
k=0
for i in range(n):
    print(i,n)
    low = min(p[i*500:i*500+500])
    if max(p[i*500:i*500+500]) - low > 50:
        k+=1
        plt.subplot(2,1,1)
        plt.plot(list(range(i*500,i*500+500)),p[i*500:i*500+500])
        plt.plot(list(range(i*500,i*500+500)),pp[i*500:i*500+500])
        plt.ylim(low*0.9995,low*1.005)
        plt.subplot(2,1,2)
        plt.plot(v[i*500:i*500+500])
        plt.show()
print(k)