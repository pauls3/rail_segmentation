
import matplotlib.pyplot as plt
import numpy as np
from csv import reader





# epochs = np.empty(len(train_iou), dtype=np.uint32)

# for i in range(len(epochs)):
# 	epochs[i] = i


# plt.plot(epochs, train_iou, label='Train')
# plt.plot(epochs, val_iou, label='Validation')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Mean IoU')
# plt.show()




# =============================================
# train_iou = []
# val_iou = []
# train_loss = []
# val_loss = []
# # epochs = 97
# epochs = np.empty(97, dtype=np.uint32)
# for i in range(97):
# 	epochs[i] = i

# with open('iou_loss.csv', 'r') as read_obj:
# 	csv_reader = reader(read_obj)
# 	header = None
# 	for row in csv_reader:
# 		label = row.pop(0)
# 		convertedRow = [float(x) for x in row]


# 		if(label == 'train_iou'):
# 			train_iou = convertedRow
# 		elif(label == 'val_iou'):
# 			val_iou = convertedRow
# 		elif(label == 'train_dice_loss'):
# 			train_loss = convertedRow
# 		elif(label == 'val_dice_loss'):
# 			val_loss = convertedRow

# plt.plot(epochs, train_iou, label='Train IoU')
# plt.plot(epochs, val_iou, label='Val IoU')
# plt.plot(epochs, train_loss, label='Train Dice Loss')
# plt.plot(epochs, val_loss, label='Val Dice Loss')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Mean IoU and Dice Loss')
# plt.show()

# =============================================

val_iou = []
val_loss = []
epochs = np.empty(43, dtype=np.uint32)
for i in range(43):
	epochs[i] = i

with open('exp_2/metrics.csv', 'r') as read_obj:
	next(read_obj)
	csv_reader = reader(read_obj)
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