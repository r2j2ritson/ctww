#----------------------------------#
#--------Batch MegaDetector--------# 
#----------------------------------#
#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly
from tqdm import tqdm
# from multiprocessing.pool import ThreadPool as workerpool
from multiprocessing.pool import Pool as workerpool

# Useful hack to force CPU inference
#
# Need to do this before any TF imports
force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#from detection.run_tf_detector import ImagePathUtils, TFDetector
#import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('tf.test.is_gpu_available:', tf.config.list_physical_devices('GPU'))


#%% Support functions for multiprocessing

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
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    results = []
    for im_file in im_files:
        results.append(process_image(im_file, tf_detector, confidence_threshold))
    return results


def process_image(im_file, tf_detector, confidence_threshold):
    """Runs the MegaDetector over a single image file.

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


#%% Main function

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

    already_processed = set([i['file'] for i in results])

    if n_cores > 1 and tf.config.list_physical_devices('GPU'):
        print('Warning: multiple cores requested, but a GPU is available; parallelization across GPUs is not currently supported, defaulting to one GPU')

    # If we're not using multiprocessing...
    if n_cores <= 1 or tf.config.list_physical_devices('GPU'):

        # Load the detector
        start_time = time.time()
        tf_detector = TFDetector(model_file)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

        # Does not count those already processed
        count = 0

        for im_file in tqdm(image_file_names):

            # Will not add additional entries not in the starter checkpoint
            if im_file in already_processed:
                print('Bypassing image {}'.format(im_file))
                continue

            count += 1

            result = process_image(im_file, tf_detector, confidence_threshold)
            results.append(result)

            # checkpoint
            if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
                print('Writing a new checkpoint after having processed {} images since last restart'.format(count))
                with open(checkpoint_path, 'w') as f:
                    json.dump({'images': results}, f)

    else:
        # when using multiprocessing, let the workers load the model
        tf_detector = model_file

        print('Creating pool with {} cores'.format(n_cores))

        if len(already_processed) > 0:
            print('Warning: when using multiprocessing, all images are reprocessed')

        pool = workerpool(int(n_cores))

        image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
        results = pool.map(partial(process_images, tf_detector=tf_detector,
                                   confidence_threshold=confidence_threshold), image_batches)

        results = list(itertools.chain.from_iterable(results))

    # results may have been modified in place, but we also return it for backwards-compatibility.
    return results


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

