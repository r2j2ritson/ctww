#' Simple MegaDetector
#'
#' Executes Microsoft's simple megadetector Python script within R using
#' 'reticulate'.Returns an output *.json file with detections for use with
#' Timelapse Image Analyser software as well as new images containing the
#' detection bounding boxes.
#'
#' @param image_path A file path to a folder containing camera trap images.
#' @param out_path A file path to a folder for storing the output images and
#'   results.json file.
#' @param detector_model A file path to a trained detector model, must end with
#'   *.pb. Microsoft's most current version-- md_v4.1.0.pb --is the internal
#'   default (detector_model = NULL).
#' @param view_when_done A boolean statement, whether to open a window to view
#'   the results in 'out_path' parameter when completed. Default is FALSE.
#'
#' @return Images with bounding box drawn and a 'results.json' file are saved in
#'   the folder defined by the 'out_path' parameter.
#' @export
#'
py_simple_megadetector <- function(
  image_path = "C:/MegaDetector/test_images/",
  out_path = "C:/MegaDetector/results/",
  detector_model = "default",
  view_when_done = FALSE){

  if(detector_model == "default"){
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
  if(reticulate::py_version() == "3.7"){
    reticulate::py_available(initialize = T)
  }else{
    pyversions <- reticulate::py_versions_windows()
    if("3.7" %in% pyversions$version){
      reticulate::use_python(python = pyversions[pyversions["version"]=="3.7",][["executable_path"]][[1]])
      reticulate::py_available(initialize = T)
    }else{stop("Python 3.7 not detected: Please install in Terminal with 'conda install python=3.7'")}
  }

  print("Load Simple MegaDetector...")
  reticulate::source_python(paste(system.file("python", package = "ctww", mustWork = T), "run_simple_megadetector.py", sep = "/"))

  print("Reticulating Simple MegaDetector on Images...")
  run_megadetector(MODEL_FILE = detector_model,
                   IMAGE_PATH = image_path,
                   OUT_PATH = out_path)

  if(view_when_done == TRUE){rstudioapi::viewer(path = out_path)}

  print("Done")
}
