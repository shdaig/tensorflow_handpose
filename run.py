import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from tensorflow.keras.models import load_model
import os
from PIL import Image
import time
from tensorflow.keras.preprocessing import image as image_utils

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
import numpy as np

from models.ColorHandPose3DNetwork import ColorHandPose3DNetwork

from utils.general import detect_keypoints, trafo_coords

def plot_hand(coords_hw, image, scale=1., xmin=0, ymin=0, color_fixed=None, linewidth=2):
    colors = [(0, 0, 130),
                       (0, 0, 200),
                       (0, 0, 245),
                       (0, 45, 255),
                       (0, 85, 255),
                       (0, 140, 255),
                       (0, 190, 255),
                       (15, 250, 240),
                       (55, 255, 190),
                       (100, 255, 145),
                       (145, 255, 100),
                       (190, 255, 55),
                       (230, 255, 15),
                       (255, 210, 0),
                       (255, 160, 0),
                       (255, 100, 0),
                       (255, 65, 0),
                       (250, 10, 0),
                       (190, 0, 0),
                       (125, 0, 0)]

    bones = [((0, 4), colors[0]),
             ((4, 3), colors[1]),
             ((3, 2), colors[2]),
             ((2, 1), colors[3]),

             ((0, 8), colors[4]),
             ((8, 7), colors[5]),
             ((7, 6), colors[6]),
             ((6, 5), colors[7]),

             ((0, 12), colors[8]),
             ((12, 11), colors[9]),
             ((11, 10), colors[10]),
             ((10, 9), colors[11]),

             ((0, 16), colors[12]),
             ((16, 15), colors[13]),
             ((15, 14), colors[14]),
             ((14, 13), colors[15]),

             ((0, 20), colors[16]),
             ((20, 19), colors[17]),
             ((19, 18), colors[18]),
             ((18, 17), colors[19])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        x1 = int(coords[0, 1] * scale) + xmin
        x2 = int(coords[1, 1] * scale) + xmin
        y1 = int(coords[0, 0] * scale) + ymin
        y2 = int(coords[1, 0] * scale) + ymin
        cv2.line(image, (x1, y1), (x2, y2), color=color, thickness=linewidth)
        
    return image

def make_prediction(img, model, sess, image_tf, hand_side_tf, evaluation, hand_scoremap_tf, image_crop_tf, scale_tf, keypoints_scoremap_tf, keypoint_coord3d_tf):
    img_height = 300
    img_width = 300
    color = (0, 255, 0)
    thickness = 2
    sx = 640 / 300
    sy = 480 / 300
    
    imgt = cv2.resize(img, dsize=(300, 300))
    imgt = cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB)
    imgt = Image.fromarray(imgt)
    imgt = image_utils.img_to_array(imgt)
    imgt = imgt.reshape(1, 300, 300, 3)

    normalize_coords = True

    y_pred = model.predict(imgt)
    y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print(y_pred_decoded[0])

    classes = ['background', 'hand']
    for box in y_pred_decoded[0]:
        xmin = int(box[-4] * sx) 
        ymin = int(box[-3] * sy) 
        xmax = int(box[-2] * sx) 
        ymax = int(box[-1] * sy)

        w = xmax - xmin
        h = ymax - ymin

        xc = xmin + w/2
        yc = ymin + h/2

        if w > h:
            h = w
        else:
            w = h

        xmin = int(xc - w/2) 
        ymin = int(yc - h/2)
        xmax = int(xc + w/2) 
        ymax = int(yc + h/2)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        frame = img[ymin:ymax, xmin:xmax]
        
        fscale = w / 256

        frame = cv2.resize(frame, dsize=(256, 256))
        frame = np.array(frame)
        frame = np.expand_dims((frame.astype('float') / 255.0) - 0.5, 0)

        hand_scoremap_v, image_crop_v, scale_v,keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, keypoints_scoremap_tf, keypoint_coord3d_tf], feed_dict={image_tf: frame})

        hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))

        img = plot_hand(coord_hw_crop, img, fscale, xmin, ymin)
        
        break

    return img

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    img_height = 300
    img_width = 300
    img_channels = 3
    intensity_mean = 127.5
    intensity_range = 127.5
    n_classes = 1
    scales = [0.08, 0.16, 0.32, 0.64,
              0.96]
    aspect_ratios = [0.5, 1.0, 2.0]
    two_boxes_for_ar1 = True
    steps = None
    offsets = None
    clip_boxes = False
    variances = [1.0, 1.0, 1.0, 1.0]
    normalize_coords = True

    model_path = 'weights/hand_weights_ssd7.h5'
    
    K.clear_session() # Clear previous models from memory.

    model = build_model(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_global=aspect_ratios,
                        aspect_ratios_per_layer=None,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range)

    model.load_weights(model_path, by_name=True)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference_ssd(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)
    
    frame_rate = 1
    prev = 0
    
    capture = cv2.VideoCapture(0)
    
    print("camera begin-----------------------------------")
    
    while capture.isOpened():
        time_elapsed = time.time() - prev
        
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        frame = frame[0:480, 0:640]
        
        if time_elapsed > 1./frame_rate:
            prev = time.time()
        
            frame = make_prediction(frame, model, sess, image_tf, hand_side_tf, evaluation, hand_scoremap_tf, image_crop_tf, scale_tf, keypoints_scoremap_tf, keypoint_coord3d_tf)
            cv2.putText(frame, f'{round(frame_rate, 2)} FPS', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
            cv2.imshow("Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("camera end-----------------------------------")
    
    capture.release()
    cv2.destroyAllWindows()
