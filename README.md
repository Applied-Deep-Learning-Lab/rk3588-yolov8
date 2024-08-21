# Yolov8 on OrangePi 5

## Configure PC for converting models to .rknn

  1. Install [**requirements**](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/requirements_cp310-2.1.0.txt

      # Install
      pip install -r requirements_cp310-2.1.0.txt
      ```

  2. Install whls for [**rknn-toolkit2**](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit2/packages/rknn_toolkit2-2.1.0+708089d1-cp310-cp310-linux_x86_64.whl

      # Install
      pip install rknn_toolkit2-2.1.0+708089d1-cp310-cp310-linux_x86_64.whl
      ```

  3. Install [**ultralytics_yolov8**](https://github.com/airockchip/ultralytics_yolov8) special for converting pt -> onnx (optimized for rknn).

      ```
      # Clone repo
      git clone https://github.com/airockchip/ultralytics_yolov8

      # Go to cloned directory
      cd ultralytics_yolov8

      # Install as package
      python setup.py install
      ```

## Convert model to .rknn
  1. Convert pt to onnx.
      ```
      from ultralyrics import YOLO
      model = YOLO("yolov8.pt")
      path = model.export(format="rknn")
      ```

  2. Convert onnx to rknn.
      ```
      # Clone repo
      git clone https://github.com/airockchip/rknn_model_zoo

      # Go to directory with converter
      cd rknn_model_zoo/examples/yolov8/python

      # Run converter
      python convert.py <path-to-onnx-model>/yolov8n.onnx rk3588 i8 ../model/yolov8n.rknn
      ```

  3. Save and send it to Orange Pi.

## Install OS

  1. Dowload image:

      | [OrangePi 5](https://drive.google.com/drive/folders/1i5zQOg1GIA4_VNGikFl2nPM0Y2MBw2M0) | [OrangePi 5B](https://drive.google.com/drive/folders/1xhP1KeW_hL5Ka4nDuwBa8N40U8BN0AC9) |
      | :---: | :---: |

  2. Burn it to SD card.

  3. Plug SD card to Orange Pi.

## Configure OrangePi for runnig models

  1. Update [**librknnrt.so**](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

      # Move to /usr/lib
      sudo mv ./librknnrt.so /usr/lib
      ```

  2. Install whls for [**rknn-toolkit-lite2**](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl

      # Install
      pip install rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl
      ```

  3. Install opencv-python and other requirements(if necessery).

      ```
      pip install opencv-python
      ```

## Run model

  1. Move rknn model to **models** directory.

  2. Change path to model in **main.py**.

  3. Run **main.py**.

      ```
      python main.py
      ```
