#' Batch MegaDetector
#'
#' Executes Microsoft's megadetector batch Python scipt within R using
#' 'reticulate'. Only returns an output *.json file with detections for use with
#' Timelapse Image Analyser software.
#'
#' @param py_virtualenv A character string naming the python virtual environment
#'   to use. Default is 'r-reticulate'.
#' @param detector_model A file path to a trained detector model, must end with
#'   *.pb. Microsoft's most current version-- md_v4.1.0.pb --is the internal
#'   default (detector_model = NULL).
#' @param image_path A file path to a folder containing camera trap images.
#' @param out_file A file path naming the output JSON file. Must end with
#'   *.json. Note that checkpoint JSON files will be saved in this file path.
#' @param threshold The confidence threshold of the detector. Must be a number
#'   between 0 and 1.0, boxes below this threshold will not be included in
#'   output JSON file. Default value is 0.1.
#' @param cp_freq Frequency of saving a temporary file 'checkpoint'. Must be an
#'   integer. Default value is -1, which disables checkpointing.
#' @param ncores Number of CPU cores to use. Must be an integer. If > 1,
#'   checkpointing will not be supported. Default value is 0.
#' @param recursive A Boolean statement whether to recurse into a directory.
#'   Default is TRUE.
#' @param resume A file path to a JSON checkpoint from which to resume the
#'   function. Must be located in the same directory as the 'out_file'
#'   parameter. Default is FALSE to ignore.
#' @param check_modules  A boolean statement, whether to check for required
#'   Python modules prior to executing function. Recommended if running for the
#'   first time on a new computer, modules will be installed if not detected.
#'   Default is FALSE.
#'
#' @return A JSON file is saved in the out_file location.
#' @export
#'
run_batch_megadetector <- function(
  py_virtualenv = "r-reticulate",
  detector_model = NULL,
  image_path = "C:/MegaDetector/test_images/",
  out_file = "C:/MegaDetector/results/results.json",
  threshold = 0.1,
  cp_freq = -1,
  ncores = 0,
  recursive = TRUE,
  resume = FALSE,
  check_modules = FALSE){

  if(is.null(detector_model)){
    detector_model <- paste(system.file("models", package = "ctww", mustWork = T), "md_v4.1.0.pb", sep = "/")
  }else if(stringr::str_detect(detector_model,pattern=".pb") == F){
    stop("Error: Invalid detector model, must be a *.pb file!")
  }
  if(length(dir(image_path)) < 1){stop("Error: No images in file path")}

  print("Initializing Python...")
  reticulate::py_available(initialize = T)
  reticulate::use_virtualenv(py_virtualenv) #test providing name as optional argument

  if(check_modules == TRUE){
    print("Check for Required Modules...")
    pkg.list <- c("tensorflow","numpy","humanfriendly","matplotlib","tqdm","requests","jsonpickle")
    lapply(pkg.list, reticulate::py_install)
    rm(pkg.list)
  }

  print("Load Microsoft MegaDetector Scripts and Utilities...")
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "ct_utils.py", sep = "/"))
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "run_tf_detector.py", sep = "/"))
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "annotation_constants.py", sep = "/"))
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "visualization_utils.py", sep = "/"))
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "run_megadetector_batch.py", sep = "/"))
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "run_batch_megadetector_main.py", sep = "/"))

  print("Reticulating Batch MegaDetector on Images...")
  run_megadetector_batch(detector_file = detector_model,
                         image_file = image_path,
                         output_file = out_file,
                         confidence_threshold = threshold,
                         checkpoint_frequency = cp_freq,
                         n_cores = ncores,
                         recurse = recursive,
                         relative = TRUE,
                         resume_from_checkpoint = resume)
  print("Done")
}
