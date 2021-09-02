import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], False)
    print(physical_devices[0])
from absl import app, flags, logging
#from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class Parameters:
    pass
FLAGS = Parameters()
FLAGS.framework = 'tf'
FLAGS.weights = './checkpoints/yolov4-416'
FLAGS.size = 416
FLAGS.tiny = False
FLAGS.model = 'yolov4'
FLAGS.image = './data/kite.jpg'
FLAGS.output = 'result_18.png'
FLAGS.iou = 0.45
FLAGS.score = 0.25

config = ConfigProto()
config.gpu_options.allow_growth = False
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = FLAGS.size
image_path = FLAGS.image

if FLAGS.framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
else:
    #saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    #infer = saved_model_loaded.signatures['serving_default']
    model = tf.keras.models.load_model(FLAGS.weights)


def get_bboxes(original_image, i=None):

    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGB)
    #if i==0 or i==1:
        #Image.fromarray(original_image).show()
    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if FLAGS.framework == 'tflite':
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        batch_data = tf.constant(images_data)
        #pred_bbox = infer(batch_data)
        pred_bbox = model.predict(batch_data)
        #print(pred_bbox)
##        if i==0 or i==1:
##            print(i, images_data)
##            print(i, infer)
##            print(pred_bbox)
##            print('---------------------')
        #for key, value in pred_bbox.items():
            #boxes = value[:, :, 0:4]
            #pred_conf = value[:, :, 4:]
        boxes = pred_bbox[:,:,:4]
        pred_conf = pred_bbox[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox
    
    
if __name__ == '__main__':
    try:
        for i in range(1):
            original_image = cv2.imread(image_path)
            pred_bbox = get_bboxes(original_image, i)
            # image = utils.draw_bbox(original_image, pred_bbox)
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            # image = utils.draw_bbox(image_data*255, pred_bbox)
            # image1 = Image.fromarray(image.astype(np.uint8))
            # image1.show()
            # cv2.imwrite(FLAGS.output[:-4]+str(i)+FLAGS.output[-4:], image)
            valid_detections =  pred_bbox[3][0]
            boxes = pred_bbox[0][0].flatten()[:valid_detections * 4]
            scores = pred_bbox[1][0][:valid_detections]
            classes = pred_bbox[2][0][:valid_detections].astype(int)
            print(valid_detections)
            print(boxes)
            print(scores)
            print(classes)
    except SystemExit:
        pass
