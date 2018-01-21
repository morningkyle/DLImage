#!/usr/bin/bash
python retrain.py \
  --bottleneck_dir=data/tf_files/bottlenecks \
  --how_many_training_steps=1000 \
  --model_dir=data/tf_files/models/ \
  --architecture="inception_v3" \
  --summaries_dir=data/tf_files/training_summaries/"inception_v3" \
  --output_graph=data/tf_files/retrained_graph.pb \
  --output_labels=data/tf_files/retrained_labels.txt \
  --image_dir=data/tf_files/flower_photos
