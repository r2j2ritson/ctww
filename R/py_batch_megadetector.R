#' Batch MegaDetector
#'
#' Executes Microsoft's megadetector batch Python scipt within R using
#' 'reticulate'. Only returns an output *.json file with detections for use with
#' Timelapse Image Analyser software.
#'
#' @param image_path A file path to a folder containing camera trap images.
#' @param out_file A file path naming the output JSON file. Must end with
#'   *.json. Note that checkpoint JSON files will be saved in this file path.
#' @param detector_model A file path to a trained detector model, must end with
#'   *.pb. Microsoft's most current version-- md_v4.1.0.pb --is the internal
#'   default (detector_model = NULL).
#' @param threshold The confidence threshold of the detector. Must be a number
#'   between 0 and 1.0, boxes below this threshold will not be included in
#'   output JSON file. Default value is 0.1.
#' @param ncores Number of CPU cores to use. Must be an integer. If > 1,
#'   checkpointing will not be supported. Default value is 0.
#' @param cp_freq Frequency of saving a temporary file 'checkpoint'. Must be an
#'   integer. Default value is -1, which disables checkpointing.
#' @param recursive A Boolean statement whether to recurse into a directory.
#'   Default is TRUE.
#' @param resume A file path to a JSON checkpoint from which to resume the
#'   function. Must be located in the same directory as the 'out_file'
#'   parameter. Default is FALSE to ignore.
#'
#' @return A JSON file is saved in the out_file location.
#' @export
#'
py_batch_megadetector <- function(
  image_path = "C:/MegaDetector/test_images/",
  out_file = "C:/MegaDetector/results/results.json",
  detector_model = "default",
  threshold = 0.1,
  ncores = 0,
  cp_freq = -1,
  recursive = TRUE,
  resume = FALSE){

  if(detector_model == "default"){
    detector_model <- paste(system.file("models", package = "ctww", mustWork = T), "md_v4.1.0.pb", sep = "/")
  }else if(stringr::str_detect(detector_model,pattern=".pb") == F){
    stop("Error: Invalid detector model, must be a *.pb file!")
  }
  if(length(dir(image_path)) < 1){stop("Error: No images in file path")}

  print("Initializing Python...")
  if(reticulate::py_version() == "3.7"){
    reticulate::py_available(initialize = T)
  }else{
    pyversions <- reticulate::py_versions_windows()
    if("3.7" %in% pyversions$version){
      reticulate::use_python(python = pyversions[pyversions["version"]=="3.7",][["executable_path"]][[1]])
      reticulate::py_available(initialize = T)
    }else{stop("Python 3.7 not detected: Please install in Terminal with 'conda install python=3.7'")}
  }

  print("Loading Batch MegaDetector...")
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "run_batch_megadetector.py", sep = "/"))

  print("Reticulating MegaDetector on Images...")
  run_megadetector_batch(detector_file = detector_model,
                         image_file = image_path,
                         output_file = out_file,
                         confidence_threshold = threshold,
                         checkpoint_frequency = cp_freq,
                         n_cores = as.integer(ncores),
                         recurse = recursive,
                         relative = TRUE,
                         resume_from_checkpoint = resume)
  print("Done")
}
