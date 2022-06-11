import pandas as pd
from tiny import create_TinyX5_model, dummy_loss
from generator import BatchGenerator
from matplotlib import pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from callbacks import CustomModelCheckpoint
from utils.utils import normalize
from utils.utils import evaluate

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

train_annot_folder = 'dataset_yolo/'
train_image_folder = 'dataset_yolo/'
valid_annot_folder = 'dataset_yolo/'
valid_image_folder = 'dataset_yolo/'

labels = ['hand']
anchors = [18,20, 30,35, 48,54, 48,54, 78,84, 134,141]
batch_size = 8
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


train_model, infer_model = create_TinyX5_model(
    nb_class            = 1,
    anchors             = anchors,
    max_box_per_image   = max_box_per_image,
    max_grid            = [416, 416],
    batch_size          = batch_size,
    warmup_batches      = 0,
    ignore_thresh       = 0.5,
    grid_scales         = [1,1],
    obj_scale           = 1,
    noobj_scale         = 1,
    xywh_scale          = 1,
    class_scale         = 1
)

optimizer = Adam(lr=1e-4, clipnorm=0.001)
train_model.compile(loss=dummy_loss, optimizer=optimizer)

early_stop = EarlyStopping(
    monitor     = 'val_loss', 
    min_delta   = 0.0, 
    patience    = 10, 
    mode        = 'min', 
    verbose     = 1
)

checkpoint = CustomModelCheckpoint(
    model_to_save   = infer_model,
    filepath        = 'checkpoints/yolov3tiny/yolov3tiny_hands_epoch-{epoch:02d}_loss-{loss:.4f}_valloss-{val_loss:.4f}.h5', 
    monitor         = 'val_loss', 
    verbose         = 1, 
    save_best_only  = True, 
    mode            = 'min', 
    period          = 1
)

reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss',
     factor=0.2,
     patience=8,
     verbose=1,
     mode = 'min',
     min_delta=0.001,
     cooldown=0,
     min_lr=0.00001
)

# callbacks = [early_stop, checkpoint, reduce_on_plateau]
callbacks = [checkpoint, reduce_on_plateau]

history = train_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=135,
    callbacks=callbacks,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    initial_epoch=0,
    workers=4,
    max_queue_size=8
)

infer_model.save_weights("hand_weights_yolov3-tiny.h5")
print("---weights saved---")
infer_model.save("hand_model_yolov3-tiny")
print("---model saved---")

plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24})
plt.savefig('history_yolov3-tiny.png')
plt.savefig('history_yolov3-tiny.pdf')
print("---plot saved---")
