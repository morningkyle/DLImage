#!/usr/bin/bash
python  classify.py \
    --graph=data/tf_files/retrained_graph.pb  \
    --labels=data/tf_files/retrained_labels.txt \
    --image=data/tf_files/flower_photos/daisy/
