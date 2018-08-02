import matplotlib.pyplot as plt
import numpy as np

directory = '/home/ljf/LJF/XXXNet/train_0.6/'
datasetLoss = 'loss_record_sgd.txt'

loss1 = []
loss2 = []
loss3 = []
step = []
i = 20

with open(directory + datasetLoss) as f3:
    for line in f3:
        l1, l2, l3= line.split()
        l1 = float(l1)
        l2 = float(l2)
        l3 = float(l3)
        loss1.append(l1)
        loss2.append(l2)
        loss3.append(l3)
        step.append(i)
        i = i + 20
f3.close()
plt.plot(step, loss1)
plt.plot(step, loss2)
#plt.plot(step, loss2, linewidth=1, linestyle="--", color="orange")
plt.plot(step, loss3)
plt.title("SGD_loss")
plt.xlabel("step")
plt.ylabel("loss_value")
plt.grid(True)
plt.show()