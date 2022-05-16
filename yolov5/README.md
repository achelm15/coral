This folder contains scripts necessary to run an edge optimized version of YOLOv5

To run on Coral: 
  - python3 detect.py --weights MODEL_PATH --source SOURCE --imgsz INPUT_SIZE --data DATASET_YAML --conf CONFIDENCE_PARAMETER\n

    MODEL_PATH - location of the model
    SOURCE - location of the source image or video
    INPUT_SIZE - one of 256, 320, 384, 448, or 512 - should match the input size of the model
    CONFIDENCE_PARAMETER - a decimal number that the model should be as confident that an object is in a location, typically anywhere from 0.25 to 0.6
