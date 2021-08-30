#---------------------------------#
#---Batch MegaDetector Function---#
#---------------------------------#

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
    
    
