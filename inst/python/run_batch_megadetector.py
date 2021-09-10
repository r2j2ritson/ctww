#----------------------------------------#
#--- Essential MegaDetector Utilities ---#
#----------------------------------------#
#%% Constants, imports, environment
import argparse
import glob
import os
import statistics
import sys
import time
import warnings
import humanfriendly
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool as workerpool
from datetime import datetime
from functools import partial
import itertools
import copy
import json
import inspect
import math
import jsonpickle
from io import BytesIO
from typing import Union
import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageFile, ImageFont, ImageDraw

#%% Set Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Turn off any GPUs
os.environ.pop('TF_CONFIG',None) #Reset TensoerFlow Configuration

#% Import and configure TensorFlow
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
tf.enable_eager_execution()
print('TensorFlow Executing Eagerly:',tf.executing_eagerly())
config = tf.ConfigProto(intra_op_parallelism_threads=os.cpu_count(),
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count= {'CPU': os.cpu_count()})
print('TensorFlow Device Count:', config.device_count)
print('TensorFlow Inter Op Parallelism Threads:', config.inter_op_parallelism_threads)
print('TensorFlow Intra Op Parallelism Threads:', config.intra_op_parallelism_threads)

#%% CT Utilities %%#
def truncate_float_array(xs, precision=3):
    """
    Vectorized version of truncate_float(...)

    Args:
    x         (list of float) List of floats to truncate
    precision (int)           The number of significant digits to preserve, should be
                              greater or equal 1
    """

    return [truncate_float(x, precision=precision) for x in xs]


def truncate_float(x, precision=3):
    """
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON

    Args:
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserve, should be
                      greater or equal 1
    """

    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor)/factor
      

def write_json(path, content, indent=1):
    with open(path, 'w') as f:
        json.dump(content, f, indent=indent)


image_extensions = ['.jpg', '.jpeg', '.gif', '.png']


def is_image_file(s):
    """
    Check a file's extension against a hard-coded set of image file extensions
    """

    ext = os.path.splitext(s)[1]
    return ext.lower() in image_extensions


def convert_xywh_to_tf(api_box):
    """
    Converts an xywh bounding box to an [y_min, x_min, y_max, x_max] box that the TensorFlow
    Object Detection API uses

    Args:
        api_box: bbox output by the batch processing API [x_min, y_min, width_of_box, height_of_box]

    Returns:
        bbox with coordinates represented as [y_min, x_min, y_max, x_max]
    """
    x_min, y_min, width_of_box, height_of_box = api_box
    x_max = x_min + width_of_box
    y_max = y_min + height_of_box
    return [y_min, x_min, y_max, x_max]


def convert_xywh_to_xyxy(api_bbox):
    """
    Converts an xywh bounding box to an xyxy bounding box.

    Note that this is also different from the TensorFlow Object Detection API coords format.
    Args:
        api_bbox: bbox output by the batch processing API [x_min, y_min, width_of_box, height_of_box]

    Returns:
        bbox with coordinates represented as [x_min, y_min, x_max, y_max]
    """

    x_min, y_min, width_of_box, height_of_box = api_bbox
    x_max, y_max = x_min + width_of_box, y_min + height_of_box
    return [x_min, y_min, x_max, y_max]


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Adapted from: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Args:
        bb1: [x_min, y_min, width_of_box, height_of_box]
        bb2: [x_min, y_min, width_of_box, height_of_box]

    These will be converted to

    bb1: [x1,y1,x2,y2]
    bb2: [x1,y1,x2,y2]

    The (x1, y1) position is at the top left corner (or the bottom right - either way works).
    The (x2, y2) position is at the bottom right corner (or the top left).

    Returns:
        intersection_over_union, a float in [0, 1]
    """

    bb1 = convert_xywh_to_xyxy(bb1)
    bb2 = convert_xywh_to_xyxy(bb2)

    assert bb1[0] < bb1[2], 'Malformed bounding box (x2 >= x1)'
    assert bb1[1] < bb1[3], 'Malformed bounding box (y2 >= y1)'

    assert bb2[0] < bb2[2], 'Malformed bounding box (x2 >= x1)'
    assert bb2[1] < bb2[3], 'Malformed bounding box (y2 >= y1)'

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0, 'Illegal IOU < 0'
    assert iou <= 1.0, 'Illegal IOU > 1'
    return iou




#%% Annotation Constants %%#
NUM_DETECTOR_CATEGORIES = 3  # this is for choosing colors, so ignoring the "empty" class

# This is the label mapping used for our incoming iMerit annotations
# Only used to parse the incoming annotations. In our database, the string name is used to avoid confusion
annotation_bbox_categories = [
    {'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'animal'},
    {'id': 2, 'name': 'person'},
    {'id': 3, 'name': 'group'},  # group of animals
    {'id': 4, 'name': 'vehicle'}
]

annotation_bbox_category_id_to_name = {}
annotation_bbox_category_name_to_id = {}

for cat in annotation_bbox_categories:
    annotation_bbox_category_id_to_name[cat['id']] = cat['name']
    annotation_bbox_category_name_to_id[cat['name']] = cat['id']

# MegaDetector outputs
detector_bbox_categories = [
    {'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'animal'},
    {'id': 2, 'name': 'person'},
    {'id': 3, 'name': 'vehicle'}
]

detector_bbox_category_id_to_name = {}
detector_bbox_category_name_to_id = {}

for cat in detector_bbox_categories:
    detector_bbox_category_id_to_name[cat['id']] = cat['name']
    detector_bbox_category_name_to_id[cat['name']] = cat['id']


#%% Visualization Utilities %%#
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}

# convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

# Retry on blob storage read failures
n_retries = 10
retry_sleep_time = 0.01
error_names_for_retry = ['ConnectionError']

def open_image(input_file: Union[str, BytesIO]) -> Image:
    """
    Opens an image in binary format using PIL.Image and converts to RGB mode.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns:
        an PIL image object in RGB mode
    """
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        try:
            response = requests.get(input_file)
        except Exception as e:
            print(f'Error retrieving image {input_file}: {e}')
            success = False
            if e.__class__.__name__ in error_names_for_retry:
                for i_retry in range(0,n_retries):
                    try:
                        time.sleep(retry_sleep_time)
                        response = requests.get(input_file)        
                    except Exception as e:
                        print(f'Error retrieving image {input_file} on retry {i_retry}: {e}')
                        continue
                    print('Succeeded on retry {}'.format(i_retry))
                    success = True
                    break
            if not success:
                raise
        try:
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise

    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L', 'I;16'):
        raise AttributeError(
            f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # Alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    #
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    #
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image

def load_image(input_file: Union[str, BytesIO]) -> Image:
    """
    Loads the image at input_file as a PIL Image into memory.

    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns: PIL.Image.Image, in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image


def resize_image(image, target_width, target_height=-1):
    """
    Resizes a PIL image object to the specified width and height; does not resize
    in place. If either width or height are -1, resizes with aspect ratio preservation.
    If both are -1, returns the original image (does not copy in this case).
    """

    # Null operation
    if target_width == -1 and target_height == -1:
        return image

    elif target_width == -1 or target_height == -1:

        # Aspect ratio as width over height
        # ar = w / h
        aspect_ratio = image.size[0] / image.size[1]

        if target_width != -1:
            # h = w / ar
            target_height = int(target_width / aspect_ratio)
        else:
            # w = ar * h
            target_width = int(aspect_ratio * target_height)

    resized_image = image.resize((target_width, target_height), Image.ANTIALIAS)
    return resized_image


def show_images_in_a_row(images):

    num = len(images)
    assert num > 0

    if isinstance(images[0], str):
        images = [Image.open(img) for img in images]

    fig, axarr = plt.subplots(1, num, squeeze=False)  # number of rows, number of columns
    fig.set_size_inches((num * 5, 25))  # each image is 2 inches wide
    for i, img in enumerate(images):
        axarr[0, i].set_axis_off()
        axarr[0, i].imshow(img)
    return fig


# The following three functions are modified versions of those at:
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

COLORS = [
    'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua', 'Azure',
    'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
    'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'Aquamarine', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def crop_image(detections, image, confidence_threshold=0.8, expansion=0):
    """
    Crops detections above *confidence_threshold* from the PIL image *image*,
    returning a list of PIL images.

    *detections* should be a list of dictionaries with keys 'conf' and 'bbox';
    see bbox format description below.  Normalized, [x,y,w,h], upper-left-origin.

    *expansion* specifies a number of pixels to include on each side of the box.
    """

    ret_images = []

    for detection in detections:

        score = float(detection['conf'])

        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection['bbox']
            ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box

            # Convert to pixels so we can use the PIL crop() function
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            if expansion > 0:
                left -= expansion
                right += expansion
                top -= expansion
                bottom += expansion

            # PIL's crop() does surprising things if you provide values outside of
            # the image, clip inputs
            left = max(left,0); right = max(right,0)
            top = max(top,0); bottom = max(bottom,0)

            left = min(left,im_width-1); right = min(right,im_width-1)
            top = min(top,im_height-1); bottom = min(bottom,im_height-1)

            ret_images.append(image.crop((left, top, right, bottom)))

        # ...if this detection is above threshold

    # ...for each detection

    return ret_images


def render_detection_bounding_boxes(detections, image,
                                    label_map={},
                                    classification_label_map={},
                                    confidence_threshold=0.8, thickness=4, expansion=0,
                                    classification_confidence_threshold=0.3,
                                    max_classifications=3):
    """
    Renders bounding boxes, label, and confidence on an image if confidence is above the threshold.

    This works with the output of the batch processing API.

    Supports classification, if the detection contains classification results according to the
    API output version 1.0.

    Args:

        detections: detections on the image, example content:
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                }
            ]

            ...where the bbox coordinates are [x, y, box_width, box_height].

            (0, 0) is the upper-left.  Coordinates are normalized.

            Supports classification results, if *detections* have the format
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                    "classifications": [
                        ["3", 0.901],
                        ["1", 0.071],
                        ["4", 0.025]
                    ]
                }
            ]

        image: PIL.Image object, output of generate_detections.

        label_map: optional, mapping the numerical label to a string name. The type of the numerical label
            (default string) needs to be consistent with the keys in label_map; no casting is carried out.

        classification_label_map: optional, mapping of the string class labels to the actual class names.
            The type of the numerical label (default string) needs to be consistent with the keys in
            label_map; no casting is carried out.

        confidence_threshold: optional, threshold above which the bounding box is rendered.
        thickness: line thickness in pixels. Default value is 4.
        expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
        classification_confidence_threshold: confidence above which classification result is retained.
        max_classifications: maximum number of classification results retained for one image.

    image is modified in place.
    """

    display_boxes = []
    display_strs = []  # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    classes = []  # for color selection

    for detection in detections:

        score = detection['conf']
        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection['bbox']
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
            clss = detection['category']
            label = label_map[clss] if clss in label_map else clss
            displayed_label = ['{}: {}%'.format(label, round(100 * score))]

            if 'classifications' in detection:

                # To avoid duplicate colors with detection-only visualization, offset
                # the classification class index by the number of detection classes
                clss = NUM_DETECTOR_CATEGORIES + int(detection['classifications'][0][0])
                classifications = detection['classifications']
                if len(classifications) > max_classifications:
                    classifications = classifications[0:max_classifications]
                for classification in classifications:
                    p = classification[1]
                    if p < classification_confidence_threshold:
                        continue
                    class_key = classification[0]
                    if class_key in classification_label_map:
                        class_name = classification_label_map[class_key]
                    else:
                        class_name = class_key
                    displayed_label += ['{}: {:5.1%}'.format(class_name.lower(), classification[1])]

            # ...if we have detection results
            display_strs.append(displayed_label)
            classes.append(clss)

        # ...if the confidence of this detection is above threshold

    # ...for each detection
    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image, display_boxes, classes,
                                 display_strs=display_strs, thickness=thickness, expansion=expansion)


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 classes,
                                 thickness=4,
                                 expansion=0,
                                 display_strs=()):
    """
    Draws bounding boxes on an image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      classes: a list of ints or strings (that can be cast to ints) corresponding to the class labels of the boxes.
             This is only used for selecting the color to render the bounding box in.
      thickness: line thickness in pixels. Default value is 4.
      expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
      display_strs: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        # print('Input must be of size [N, 4], but is ' + str(boxes_shape))
        return  # no object detection on this image, return
    for i in range(boxes_shape[0]):
        if display_strs:
            display_str_list = display_strs[i]
            draw_bounding_box_on_image(image,
                                       boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                                       classes[i],
                                       thickness=thickness, expansion=expansion,
                                       display_str_list=display_str_list)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               clss=None,
                               thickness=4,
                               expansion=0,
                               display_str_list=(),
                               use_normalized_coordinates=True,
                               label_font_size=16):
    """
    Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box - upper left.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    clss: str, the class of the object in this bounding box - will be cast to an int.
    thickness: line thickness. Default value is 4.
    expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
    display_str_list: list of strings to display in box
        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    label_font_size: font size to attempt to load arial.ttf with
    """
    if clss is None:
        color = COLORS[1]
    else:
        color = COLORS[int(clss) % len(COLORS)]

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    if expansion > 0:
        left -= expansion
        right += expansion
        top -= expansion
        bottom += expansion

        # Deliberately trimming to the width of the image only in the case where
        # box expansion is turned on.  There's not an obvious correct behavior here,
        # but the thinking is that if the caller provided an out-of-range bounding
        # box, they meant to do that, but at least in the eyes of the person writing
        # this comment, if you expand a box for visualization reasons, you don't want
        # to end up with part of a box.
        #
        # A slightly more sophisticated might check whether it was in fact the expansion
        # that made this box larger than the image, but this is the case 99.999% of the time
        # here, so that doesn't seem necessary.
        left = max(left,0); right = max(right,0)
        top = max(top,0); bottom = max(bottom,0)

        left = min(left,im_width-1); right = min(right,im_width-1)
        top = min(top,im_height-1); bottom = min(bottom,im_height-1)

    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype('arial.ttf', label_font_size)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)

        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)

        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)

        text_bottom -= (text_height + 2 * margin)


def render_iMerit_boxes(boxes, classes, image,
                        label_map=annotation_bbox_category_id_to_name):
    """
    Renders bounding boxes and their category labels on a PIL image.

    Args:
        boxes: bounding box annotations from iMerit, format is [x_rel, y_rel, w_rel, h_rel] (rel = relative coords)
        classes: the class IDs of the predicted class of each box/object
        image: PIL.Image object to annotate on
        label_map: optional dict mapping classes to a string for display

    Returns:
        image will be altered in place
    """

    display_boxes = []
    display_strs = []  # list of list, one list of strings for each bounding box (to accommodate multiple labels)
    for box, clss in zip(boxes, classes):
        if len(box) == 0:
            assert clss == 5
            continue
        x_rel, y_rel, w_rel, h_rel = box
        ymin, xmin = y_rel, x_rel
        ymax = ymin + h_rel
        xmax = xmin + w_rel

        display_boxes.append([ymin, xmin, ymax, xmax])

        if label_map:
            clss = label_map[int(clss)]
        display_strs.append([clss])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)


def render_megadb_bounding_boxes(boxes_info, image):
    """
    Args:
        boxes_info: list of dict, each dict represents a single detection
            {
                "category": "animal",
                "bbox": [
                    0.739,
                    0.448,
                    0.187,
                    0.198
                ]
            }
            where bbox coordinates are normalized [x_min, y_min, width, height]
        image: PIL.Image.Image, opened image
    """
    display_boxes = []
    display_strs = []
    classes = []  # ints, for selecting colors

    for b in boxes_info:
        x_min, y_min, w_rel, h_rel = b['bbox']
        y_max = y_min + h_rel
        x_max = x_min + w_rel
        display_boxes.append([y_min, x_min, y_max, x_max])
        display_strs.append([b['category']])
        classes.append(detector_bbox_category_name_to_id[b['category']])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)


def render_db_bounding_boxes(boxes, classes, image, original_size=None,
                             label_map=None, thickness=4, expansion=0):
    """
    Render bounding boxes (with class labels) on [image].  This is a wrapper for
    draw_bounding_boxes_on_image, allowing the caller to operate on a resized image
    by providing the original size of the image; bboxes will be scaled accordingly.
    """

    display_boxes = []
    display_strs = []

    if original_size is not None:
        image_size = original_size
    else:
        image_size = image.size

    img_width, img_height = image_size

    for box, clss in zip(boxes, classes):

        x_min_abs, y_min_abs, width_abs, height_abs = box

        ymin = y_min_abs / img_height
        ymax = ymin + height_abs / img_height

        xmin = x_min_abs / img_width
        xmax = xmin + width_abs / img_width

        display_boxes.append([ymin, xmin, ymax, xmax])

        if label_map:
            clss = label_map[int(clss)]
        display_strs.append([str(clss)])  # need to be a string here because PIL needs to iterate through chars

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs,
                                 thickness=thickness, expansion=expansion)


def draw_bounding_boxes_on_file(input_file, output_file, detections, confidence_threshold=0.0,
                                detector_label_map=DEFAULT_DETECTOR_LABEL_MAP):
    """
    Render detection bounding boxes on an image loaded from file, writing the results to a
    new images file.  "detections" is in the API results format.
    """
    
    image = open_image(input_file)

    render_detection_bounding_boxes(
            detections, image, label_map=detector_label_map,
            confidence_threshold=confidence_threshold)

    image.save(output_file)


#%% Classes %%#
class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings

class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    the MegaDetector (TF). The inference batch size is set to 1; code needs to be modified
    to support larger batch sizes, including resizing appropriately.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # An enumeration of failure reasons
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '3': 'vehicle'  # available in megadetector v4+
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads model from model_path and starts a tf.Session with this graph. Obtains
        input and output tensor handles."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(config=config,graph=detection_graph) #add configuration and graph
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(tf_coords):
        """Converts coordinates from the model's output format [y1, x1, y2, x2] to the
        format used by our API and MegaDB: [x1, y1, width, height]. All coordinates
        (including model outputs) are normalized in the range [0, 1].

        Args:
            tf_coords: np.array of predicted bounding box coordinates from the TF detector,
                has format [y1, x1, y2, x2]

        Returns: list of Python float, predicted bounding box coordinates [x1, y1, width, height]
        """
        # change from [y1, x1, y2, x2] to [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """From [x1, y1, width, height] to [y1, x1, y2, x2], where x1 is x_min, x2 is x_max

        This is an extraneous step as the model outputs [y1, x1, y2, x2] but were converted to the API
        output format - only to keep the interface of the sync API.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.

        Args:
            model_path: .pb file of the model.

        Returns: the loaded graph.
        """
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id,
                                      detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.

        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes == b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result


#%% Support functions for multiprocessing %%#
def process_images(im_files, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a list of image files.

    Args
    - im_files: list of str, paths to image files
    - tf_detector: TFDetector (loaded model) or str (path to .pb model file)
    - confidence_threshold: float, only detections above this threshold are returned

    Returns
    - results: list of dict, each dict represents detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    if isinstance(tf_detector, str):
        start_time = time.time()
        tf_detector = TFDetector(tf_detector)
        elapsed = time.time() - start_time
        #print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    results = []
    for im_file in im_files:
        results.append(process_image(im_file, tf_detector, confidence_threshold))
    return results


def process_image(im_file, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a single image file. Modified for multiprocessing...

    Args
    - im_file: str, path to image file
    - tf_detector: TFDetector, loaded model
    - confidence_threshold: float, only detections above this threshold are returned

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    print('Processing image {}'.format(im_file))
    try:
        tf_detector = TFDetector(tf_detector)      
        image = load_image(im_file)
    except Exception as e:
        print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_IMAGE_OPEN
        }
        return result

    try:
        result = tf_detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold)
    except Exception as e:
        print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': TFDetector.FAILURE_TF_INFER
        }
        return result

    return result


def chunks_by_number_of_chunks(ls, n):
    """Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]

#%% Load and Run Detector 
def load_and_run_detector_batch(model_file, image_file_names, checkpoint_path=None,
                                confidence_threshold=0, checkpoint_frequency=-1,
                                results=None, n_cores=0):
    """
    Args
    - model_file: str, path to .pb model file
    - image_file_names: list of str, paths to image files
    - checkpoint_path: str, path to JSON checkpoint file
    - confidence_threshold: float, only detections above this threshold are returned
    - checkpoint_frequency: int, write results to JSON checkpoint file every N images
    - results: list of dict, existing results loaded from checkpoint
    - n_cores: int, # of CPU cores to use

    Returns
    - results: list of dict, each dict represents detections on one image
    """
    n_cores = int(round(n_cores))
    if results is None:
        results = []
        
    if n_cores <= 1:
      # Load the detector
        start_time = time.time()
        tf_detector = TFDetector(model_file)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))
        
        for im_file in tqdm(image_file_names):
          result = process_image(im_file, tf_detector, confidence_threshold)
          results.append(result)
            
    else:
      tf_detector = model_file
      
      print('Creating pool with {} cores'.format(n_cores))
      pool = workerpool(int(n_cores))
      
      image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
      results = pool.map(partial(process_images, tf_detector=tf_detector, confidence_threshold=confidence_threshold), image_batches)
      results = list(itertools.chain.from_iterable(results))
    
      #results = [pool.apply(process_images, args=c(im_file, tf_detector, confidence_threshold)) for im_file in image_file_names]
      #results = pool.starmap(process_image, [(im_file, tf_detector, confidence_threshold) for im_file in image_batches])
      #results = list(itertools.chain.from_iterable(results))
      pool.close()

    return results


#%% Write Output JSON file
def write_results_to_file(results, output_file, relative_path_base=None):
    """Writes list of detection results to JSON output file. Format matches
    https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

    Args
    - results: list of dict, each dict represents detections on one image
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))

#%% Main Function
def run_megadetector_batch(detector_file, image_file, output_file, confidence_threshold=0,
                           checkpoint_frequency=-1, n_cores=0, recurse=True, relative=True,
                           resume_from_checkpoint=False):
  
    assert os.path.exists(detector_file), 'Specified detector_file does not exist'
    assert 0.0 < confidence_threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison
    assert output_file.endswith('.json'), 'output_file specified needs to end with .json'
    if checkpoint_frequency != -1:
        assert checkpoint_frequency > 0, 'Checkpoint_frequency needs to be > 0 or == -1'
    if relative:
        assert os.path.isdir(image_file), 'image_file must be a directory when relative is set'
    if os.path.exists(output_file):
        print('Warning: output_file {} already exists and will be overwritten'.format(output_file))

    # Load the checkpoint if available #
    ## Relative file names are only output at the end; all file paths in the checkpoint are still full paths.
    if resume_from_checkpoint:
        assert os.path.exists(resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'
        with open(resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'The file saved as checkpoint does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []

    # Find the images to score; images can be a directory, may need to recurse
    if os.path.isdir(image_file):
        image_file_names = ImagePathUtils.find_images(image_file, recursive=recurse)
        print('{} image files found in the input directory'.format(len(image_file_names)))
    # A json list of image paths
    elif os.path.isfile(image_file) and image_file.endswith('.json'):
        with open(image_file) as f:
            image_file_names = json.load(f)
        print('{} image files found in the json list'.format(len(image_file_names)))
    # A single image file
    elif os.path.isfile(image_file) and ImagePathUtils.is_image_file(image_file):
        image_file_names = [image_file]
        print('A single image at {} is the input file'.format(image_file))
    else:
        raise ValueError('image_file specified is not a directory, a json list, or an image file, '
                         '(or does not have recognizable extensions).')

    assert len(image_file_names) > 0, 'Specified image_file does not point to valid image files'
    assert os.path.exists(image_file_names[0]), 'The first image to be scored does not exist at {}'.format(image_file_names[0])

    output_dir = os.path.dirname(output_file)

    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
        
    assert not os.path.isdir(output_file), 'Specified output file is a directory'

    # Test that we can write to the output_file's dir if checkpointing requested
    if checkpoint_frequency != -1:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': []}, f)
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
    else:
        checkpoint_path = None

    start_time = time.time()
    
    n_cores = int(round(n_cores))

    results = load_and_run_detector_batch(model_file=detector_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=confidence_threshold,
                                          checkpoint_frequency=checkpoint_frequency,
                                          results=results,
                                          n_cores=n_cores)

    elapsed = time.time() - start_time
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))

    relative_path_base = None
    if relative:
        relative_path_base = image_file
    write_results_to_file(results, output_file, relative_path_base=relative_path_base)

    if checkpoint_path:
        os.remove(checkpoint_path)
        print('Deleted checkpoint file')

    print('Done!')



