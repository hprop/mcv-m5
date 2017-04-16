import os
import sys,time
import tempfile
import collections
import math

import numpy as np
import cv2

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from models.yolo import build_yolo
from tools.yolo_utils import *
from models import ssd
from tools import ssd_utils

# Input parameters to select the Dataset and the model used
dataset_name = 'TT100K_detection' #set to Udacity dataset otherwise
model_name = 'ssd300' #options: 'yolo', 'tiny-yolo', 'ssd300',
                            #'ssd300_pretrained' 'ssd_resnet50'

# Input parameters to perform data preprocessing
samplewise_center = False
samplewise_std_normalization = False

# Save results in a temporary directory, distributed by size (small, medium,
# big)
save_results = True
if save_results:
  save_path = tempfile.mkdtemp(prefix='eval-detection-results-')
  os.mkdir(os.path.join(save_path, 'small'))
  os.mkdir(os.path.join(save_path, 'medium'))
  os.mkdir(os.path.join(save_path, 'big'))
  print('Saving results in', save_path)
else:
  save_path = None

if len(sys.argv) < 3:
  print "USAGE: python eval_detection_fscore.py weights_file path_to_images"
  quit()

if dataset_name == 'TT100K_detection':
    classes = ['i2','i4','i5','il100','il60','il80','io','ip','p10','p11',
               'p12','p19','p23','p26','p27','p3','p5','p6','pg','ph4','ph4.5',
               'ph5','pl100','pl120','pl20','pl30','pl40','pl5','pl50','pl60','pl70',
               'pl80','pm20','pm30','pm55','pn','pne','po','pr40','w13','w32','w55',
               'w57','w59','wo']
elif dataset_name == 'Udacity':
    classes = ['Car','Pedestrian','Truck']
else:
    print "Error: Dataset not found!"
    quit()

NUM_CLASSES = len(classes)
priors = [[0.9,1.2], [1.05,1.35], [2.15,2.55], [3.25,3.75], [5.35,5.1]]

if model_name in ['yolo', 'tiny-yolo']:
  input_shape = (3, 320, 320)
  HEIGHT, WIDTH  = input_shape[1:]
  NUM_PRIORS  = len(priors)
  img_channel_axis = 0
  tiny_yolo = (model_name == 'tiny-yolo')
  model = build_yolo(img_shape=input_shape,n_classes=NUM_CLASSES, n_priors=5,
                     load_pretrained=False,freeze_layers_from='base_model',
                     tiny=tiny_yolo)

elif model_name == 'ssd300':
  input_shape = (320, 320, 3)
  detection_threshold = 0.5 # Min probablity for a prediction to be considered
  nms_threshold       = 0.5 # Non Maximum Suppression threshold
  HEIGHT, WIDTH  = input_shape[:2]
  img_channel_axis = 2
  model = ssd.build_ssd300(input_shape, NUM_CLASSES + 1)
  ssd_utils.initialize_module(model, input_shape, NUM_CLASSES + 1,
                              overlap_threshold=0.5, nms_thresh=nms_threshold,
                              top_k=40)

elif model_name == 'ssd300_pretrained':
  input_shape = (320, 320, 3)
  detection_threshold = 0.5 # Min probablity for a prediction to be considered
  nms_threshold       = 0.5 # Non Maximum Suppression threshold
  HEIGHT, WIDTH  = input_shape[:2]
  img_channel_axis = 2
  model = ssd.build_ssd300_pretrained(input_shape, NUM_CLASSES + 1)
  ssd_utils.initialize_module(model, input_shape, NUM_CLASSES + 1,
                              overlap_threshold=0.5, nms_thresh=nms_threshold,
                              top_k=40)

elif model_name == 'ssd_resnet50':
  input_shape = (320, 320, 3)
  detection_threshold = 0.4 # Min probablity for a prediction to be considered
  nms_threshold       = 0.3 # Non Maximum Suppression threshold
  HEIGHT, WIDTH  = input_shape[:2]
  img_channel_axis = 2
  model = ssd.build_ssd_resnet50(input_shape, NUM_CLASSES + 1)
  ssd_utils.initialize_module(model, input_shape, NUM_CLASSES + 1,
                              overlap_threshold=0.2, nms_thresh=nms_threshold,
                              top_k=2)

model.load_weights(sys.argv[1])

test_dir = sys.argv[2]
imfiles = [os.path.join(test_dir,f) for f in os.listdir(test_dir)
                                    if os.path.isfile(os.path.join(test_dir,f))
                                    and f.endswith('jpg')]

if len(imfiles) == 0:
  print "ERR: path_to_images do not contain any jpg file"
  quit()

inputs = []
img_paths = []
chunk_size = 128 # we are going to process all image files in chunks

ok = 0.
total_true = 0.
total_pred = 0.
total_img = 0
total_time = 0.

analysis_small = {'hit': 0, 'real': 0, 'pred': 0, 'files': []}
analysis_medium = {'hit': 0, 'real': 0, 'pred': 0, 'files': []}
analysis_big = {'hit': 0, 'real': 0, 'pred': 0, 'files': []}

def classify_box(b):
  if b.h <= 0.25:
    return 'small'
  elif b.h <= 0.50:
    return 'medium'
  else:
    return 'big'

for i, img_path in enumerate(imfiles):
  img = image.load_img(img_path, target_size=(HEIGHT, WIDTH))
  img = image.img_to_array(img)
  img = img / 255.
  if samplewise_center:
     img -= np.mean(img, axis=img_channel_axis, keepdims=True)
  if samplewise_std_normalization:
     img /= (np.std(img, axis=img_channel_axis, keepdims=True) + 1e-7)

  inputs.append(img.copy())
  img_paths.append(img_path)

  if len(img_paths)%chunk_size == 0 or i+1 == len(imfiles):
    inputs = np.array(inputs)
    start_time = time.time()
    net_out = model.predict(inputs, batch_size=16, verbose=1)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    total_img += len(img_paths)
    print ('{} images predicted in {:.5f} seconds. {:.5f} fps').format(
      len(inputs), elapsed_time, len(inputs) / elapsed_time)

    # find correct detections (per image)
    for i,img_path in enumerate(img_paths):
        if model_name in ['yolo', 'tiny-yolo']:
          boxes_pred = yolo_postprocess_net_out(net_out[i], priors, classes,
                                                detection_threshold, nms_threshold)
        else:
          boxes_pred = ssd_utils.ssd_postprocess_prediction(
            net_out[i], len(classes), detection_threshold)

        boxes_true = []
        label_path = img_path.replace('jpg','txt')
        gt = np.loadtxt(label_path)
        if len(gt.shape) == 1:
          gt = gt[np.newaxis,]
        for j in range(gt.shape[0]):
          bx = BoundBox(len(classes))
          bx.probs[int(gt[j,0])] = 1.
          bx.x, bx.y, bx.w, bx.h = gt[j,1:].tolist()
          boxes_true.append(bx)

        total_true += len(boxes_true)
        true_matched = np.full(len(boxes_true), -1)

        # boxes_pred: list of BBox objects (x, y, w, h, c=score,
        #                                   probs=array[0..44])
        for i, b in enumerate(boxes_pred):
          # discard if detection is lower than `detection_threshold`
          if b.probs[np.argmax(b.probs)] < detection_threshold:
             continue
          total_pred += 1.

          # compare with real bboxes
          for t, a in enumerate(boxes_true):
            if true_matched[t] >= 0:
              continue
            if box_iou(a, b) > 0.5 and np.argmax(a.probs) == np.argmax(b.probs):
              true_matched[t] = i
              ok += 1.
              break

        for i, b in enumerate(boxes_true):
          size = classify_box(b)
          if size == 'small':
            analysis_small['real'] += 1
            analysis_small['hit'] += 1 if (true_matched[i] >= 0) else 0
            analysis_small['pred'] += 1 if (true_matched[i] >= 0) else 0
            analysis_small['files'].append(img_path)

          elif size == 'medium':
            analysis_medium['real'] += 1
            analysis_medium['hit'] += 1 if (true_matched[i] >= 0) else 0
            analysis_medium['pred'] += 1 if (true_matched[i] >= 0) else 0
            analysis_medium['files'].append(img_path)

          elif size == 'big':
            analysis_big['real'] += 1
            analysis_big['hit'] += 1 if (true_matched[i] >= 0) else 0
            analysis_big['pred'] += 1 if (true_matched[i] >= 0) else 0
            analysis_big['files'].append(img_path)

          if save_path:
            dst = os.path.join(save_path, size, os.path.basename(img_path))
            im = cv2.imread(img_path)
            im = yolo_draw_detections(boxes_pred, im, priors, classes,
                                      detection_threshold, nms_threshold)
            cv2.imwrite(dst, im)

          # TODO: save result
          # # You can visualize/save per image results with this:
        # im = cv2.imread(img_path)
        # im = yolo_draw_detections(boxes_pred, im, priors, classes, detection_threshold, nms_threshold)
        # cv2.imwrite('/tmp/detection_result.png', im)
        # raw_input('Press Enter to continue...')


        # false positives counting
        false_neg = [i for i in range(len(boxes_pred)) if i not in true_matched]
        for i in false_neg:
          size = classify_box(boxes_pred[i])
          if size == 'small':
            analysis_small['pred'] += 1
          if size == 'medium':
            analysis_medium['pred'] += 1
          if size == 'big':
            analysis_big['pred'] += 1

    inputs = []
    img_paths = []

    #print 'total_true:',total_true,' total_pred:',total_pred,' ok:',ok
    p = 0. if total_pred == 0 else (ok/total_pred)
    r = ok/total_true
    print('Precission = '+str(p))
    print('Recall     = '+str(r))
    f = 0. if (p+r) == 0 else (2*p*r/(p+r))
    print('F-score    = '+str(f))


print('Average FPS: {:.5f}'.format(total_img/total_time))

def print_stats(analysis, size):
  if analysis['pred'] == 0:
    prec = 0
  else:
    prec = float(analysis['hit']) / analysis['pred']

  rec = float(analysis['hit']) / analysis['real'] if analysis['real'] <> 0 else 0
  fsco = 0. if (prec + rec) == 0 else float(2 * prec * rec) / (prec + rec)

  print('Analysis {}: Prec {:.5f}, Rec {:.5f}, F1 {:.5f}, Total {}'.format(
    size, prec, rec, fsco, analysis['real']))

print_stats(analysis_small, 'small')
print_stats(analysis_medium, 'medium')
print_stats(analysis_big, 'big')
