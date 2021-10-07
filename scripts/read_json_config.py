
import os
import json


def main():
	config_fn = '/home/paul/datasets/mapillary_vistas/config_v2.0.json'
	with open(config_fn) as config_file:
		config = json.load(config_file)
	config_labels = config['labels']

	# calculate label color mapping
	colormap = []
	for i in range(0, len(config_labels)):
	    colormap = colormap + config_labels[i]['color']
	    name = config_labels[i]['name']
	    # name = name.replace(' ', '_')
	    print(name)




if __name__== "__main__":
  main()