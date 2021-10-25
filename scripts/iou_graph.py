
from os import read
import matplotlib.pyplot as plt
import numpy as np
from csv import reader


val_iou = []
val_loss = []
epochs = np.empty(154, dtype=np.uint32)
for i in range(154):
	epochs[i] = i

with open('../exp_3/metrics.csv', 'r') as read_obj:
	# next(read_obj)
	csv_reader = reader(read_obj)
	# next(read_obj)
	header = None
	for row in csv_reader:
		loss = float(row[2])
		miou = float(row[4])

		val_loss.append(loss)
		val_iou.append(miou)

plt.plot(epochs, val_iou, label='Val IoU')
plt.plot(epochs, val_loss, label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean IoU and Loss')
plt.show()