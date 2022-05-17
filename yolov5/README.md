This folder contains scripts necessary to run an edge optimized version of YOLOv5

To run on Coral: 
  ```bash
  python3 detect.py --weights MODEL_PATH --source SOURCE --imgsz INPUT_SIZE --data DATASET_YAML --conf CONFIDENCE_PARAMETER
  ```
  
   - MODEL_PATH - location of the model
   - SOURCE - location of the source image or video
   - INPUT_SIZE - one of 256, 320, 384, 448, or 512 - should match the input size of the model
   - CONFIDENCE_PARAMETER - a decimal number that the model should be as confident that an object is in a location, typically anywhere from 0.25 to 0.6


* The yaml file should contain the same list as found in the yaml file used to train the model

- The current detect.py uses NumPy arrays, however a different version exists in a branch entitled "torch" which uses torch tensors. This may work faster on the Raspberry Pi with Coral accelerator.
- The current detect.py may appear slow, however that is due only to OpenCV. If you remove the write or imshow statements, the script will run much faster. Print statements exist to demonstrate which parts of the code are taking a long or short time.

- general.py contains auxilary function necessary for detect.py to work properly.
