#!/usr/bin/bash
python  label_image.py \
    --graph=data/tf_files/retrained_graph.pb  \
    --labels=data/tf_files/retrained_labels.txt \
    --image=data/tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
