import json

config_fn = 'rs19-config.json'
id_to_ignore_or_group = {}
color_mapping = []
id_to_trainid = {}



# load railsem19 config
with open(config_fn) as config_file:
    config = json.load(config_file)
config_labels = config['labels']

# calculate label color mapping
colormap = []
id2name = {}
for i in range(0, len(config_labels)):
    colormap = colormap + config_labels[i]['color']
    # id2name[i] = config_labels[i]['readable']
    name = config_labels[i]['name']
    name = name.replace(' ', '_')
    id2name[i] = name
color_mapping = colormap


for ii in id2name:
    print(ii)