
from os import read
import matplotlib.pyplot as plt
import numpy as np
from csv import reader


val_iou = []
val_loss = []
epochs = []

ii = 0
with open('../logs/sfsegnets_log_2022_01_04_13_03_45_rank_0.log', 'r') as file0:
	for line in file0:
		if 'best record:' not in line and '[val loss' in line and '[mean_iu' in line:
			index0 = line.find('[val loss ')
			index1 = line.find(']', index0)

			index2 = line.find('[mean_iu ')
			index3 = line.find(']', index2)

			loss = float(line[index0 + 9 : index1])
			iou = float(line[index2 + 8 : index3])

			val_iou.append(iou)
			val_loss.append(loss)
			epochs.append(ii)

			ii = ii + 1

plt.plot(epochs, val_iou, label='Val IoU')
plt.plot(epochs, val_loss, label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean IoU and Loss')
plt.show()