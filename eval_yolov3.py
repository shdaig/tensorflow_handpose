import pandas as pd
from yolo import create_yolov3_model, dummy_loss
from generator_yolov3 import BatchGenerator
from matplotlib import pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from callbacks import CustomModelCheckpoint
from utils.utils import normalize
from utils.utils import evaluate
from keras.models import load_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_annot_folder = 'dataset_hands/dataset_hands_yolo/train/'
train_image_folder = 'dataset_hands/dataset_hands_yolo/train/'
valid_annot_folder = 'dataset_hands/dataset_hands_yolo/val/'
valid_image_folder = 'dataset_hands/dataset_hands_yolo/val/'

labels = ['hand']
anchors = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
batch_size = 2
min_input_size = 320
max_input_size = 608

def parse_csv_annotation(ann_file, img_dir, cache_name, labels=[]):
    all_insts = []
    seen_labels = {}
    
    last_file = ''
    flag = False
    
    annotations = pd.read_csv(img_dir + ann_file, delimiter=',')
    rows = annotations.size // 8
    
    i = 0
    while i < rows:
        img = {'object': []}
        img['filename'] = img_dir + annotations.filename[i]
        img['width'] = annotations.width[i]
        img['height'] = annotations.height[i]
        j = i
        filename = annotations.filename[i]
        while (j < rows) and (filename == annotations.filename[j]):
            obj = {}
            obj['name'] = 'hand'
            if obj['name'] in seen_labels:
                seen_labels[obj['name']] += 1
            else:
                seen_labels[obj['name']] = 1
            obj['xmin'] = annotations.xmin[j]
            obj['ymin'] = annotations.ymin[j]
            obj['xmax'] = annotations.xmax[j]
            obj['ymax'] = annotations.ymax[j]
            img['object'] += [obj]
            j += 1;
        i = j
        all_insts += [img]   
    return all_insts, seen_labels

def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels
):
    train_ints, train_labels = parse_csv_annotation('train_labels.csv', train_image_folder, train_cache, labels)
    valid_ints, valid_labels = parse_csv_annotation('val_labels.csv', valid_image_folder, valid_cache, labels)
    
    print(len(train_ints))
    print(len(valid_ints))

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])
    
    return train_ints, valid_ints, max_box_per_image

train_ints, valid_ints, max_box_per_image = create_training_instances(
    train_annot_folder,
    train_image_folder,
    '',
    valid_annot_folder,
    valid_image_folder,
    '',
    labels
)

train_generator = BatchGenerator(
    instances           = train_ints, 
    anchors             = anchors,   
    labels              = labels,        
    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image   = max_box_per_image,
    batch_size          = batch_size,
    min_net_size        = min_input_size,
    max_net_size        = max_input_size,   
    shuffle             = True, 
    jitter              = 0.1, 
    norm                = normalize
)

valid_generator = BatchGenerator(
    instances           = valid_ints, 
    anchors             = anchors,   
    labels              = labels,        
    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image   = max_box_per_image,
    batch_size          = batch_size,
    min_net_size        = min_input_size,
    max_net_size        = max_input_size,   
    shuffle             = True, 
    jitter              = 0.0, 
    norm                = normalize
)

# model_path = 'hand_model_yolov3'
# infer_model = load_model(model_path)
# infer_model.summary()
train_model, infer_model = create_yolov3_model(
    nb_class            = 1,
    anchors             = anchors,
    max_box_per_image   = max_box_per_image,
    max_grid            = [416, 416],
    batch_size          = batch_size,
    warmup_batches      = 0,
    ignore_thresh       = 0.5,
    grid_scales         = [1,1,1],
    obj_scale           = 5,
    noobj_scale         = 1,
    xywh_scale          = 1,
    class_scale         = 1
)

infer_model.load_weights('hand_weights_yolov3.h5')

average_precisions, recalls, precisions = evaluate(infer_model, valid_generator, iou_threshold=0.5, obj_thresh=0.5, nms_thresh=0.2, net_h=416, net_w=416, save_path='result')

# print the score
for label, average_precision in average_precisions.items():
    print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
    print('recalls:\t', recalls)
    print('precision:\t', precisions)
