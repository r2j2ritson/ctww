#' Simple MegaDetector
#'
#' Executes Microsoft's simple megadetector Python script within R using
#' 'reticulate'.Returns an output *.json file with detections for use with
#' Timelapse Image Analyser software as well as new images containing the
#' detection bounding boxes.
#'
#' @param py_virtualenv A character string naming the python virtual environment
#'   to use. Default is 'r-reticulate'.
#' @param detector_model A file path to a trained detector model, must end with
#'   *.pb. Microsoft's most current version-- md_v4.1.0.pb --is the internal
#'   default (detector_model = NULL).
#' @param image_path A file path to a folder containing camera trap images.
#' @param out_path A file path to a folder for storing the output images and
#'   results.json file.
#' @param view_when_done A boolean statement, whether to open a window to view
#'   the results in 'out_path' parameter when completed. Default is FALSE.
#' @param check_modules A boolean statement, whether to check for required
#'   Python modules prior to executing function. Recommended if running for the
#'   first time on a new computer, modules will be installed if not detected.
#'   Default is FALSE.
#'
#' @return Images with bounding box drawn and a 'results.json' file are saved in
#'   the folder defined by the 'out_path' parameter.
#' @export
#'
run_simple_megadetector <- function(
  py_virtualenv = "r-reticulate",
  detector_model = NULL,
  image_path = "C:/MegaDetector/test_images/",
  out_path = "C:/MegaDetector/results/",
  view_when_done = FALSE,
  check_modules = FALSE){

  if(is.null(detector_model)){
    detector_model <- paste(system.file("models", package = "ctww", mustWork = T), "md_v4.1.0.pb", sep = "/")
  }else if(stringr::str_detect(detector_model,pattern=".pb") == F){
    stop("Error: Invalid detector model, must be a *.pb file!")
  }
  if(length(dir(image_path)) < 1){stop("Error: No images in file path")}
  if(dir.exists(out_path) == F){
    print("Out Path Does Not Exist --> Creating file path")
    dir.create(out_path)
  }

  print("Initializing Python...")
  reticulate::py_available(initialize = T)
  reticulate::use_virtualenv(py_virtualenv)

  if(check_modules == TRUE){
    print("Check for Required Modules...")
    pkg.list <- c("tensorflow","numpy","humanfriendly","matplotlib","tqdm","requests","jsonpickle")
    lapply(pkg.list, reticulate::py_install)
    rm(pkg.list)
  }

  print("Load Microsoft MegaDetector Scripts and Utilities...")
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "run_megadetector.py", sep = "/"))

  print("Reticulating Simple MegaDetector on Images...")
  run_megadetector(MODEL_FILE = detector_model,
                   IMAGE_PATH = image_path,
                   OUT_PATH = out_path)

  if(view_when_done == TRUE){rstudioapi::viewer(out_path)}

  print("Done")
}
