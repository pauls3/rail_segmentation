import re


pattern_text_train = r"(?<=train:	{'dice_loss': )(.*)(?= 'iou_score')"
pattern_train = re.compile(pattern_text_train)
pattern_text_val = r"(?<=val:	{'dice_loss': )(.*)(?= 'iou_score')"
pattern_val = re.compile(pattern_text_val)

file1 = open('epoch_info.txt', 'r')
Lines = file1.readlines()


export_train = ''
export_val = ''

train = ''
val = ''
for line in Lines:

	# train = pattern_train.match(line)
	# val = pattern_val.match(line)
	train = re.search("(?<=train:	{'dice_loss': )(.*)(?= 'iou_score')", line)
	val = re.search("(?<=val:	{'dice_loss': )(.*)(?= 'iou_score')", line)

	if train is not None:
		export_train = export_train + train.group()

	if val is not None:
		export_val = export_val + val.group()

print(export_train)

print('\n\n\n')

print(export_val)
	