# Yolov8 on OrangePi 5

## Configure PC for converting models to .rknn

  1. Install [**requirements**](https://github.com/airockchip/rknn-toolkit2/tree/v2.1.0/rknn-toolkit2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknn-toolkit2/packages/requirements_cp310-2.1.0.txt

      # Install
      pip install -r requirements_cp310-2.1.0.txt
      ```

  2. Install whls for [**rknn-toolkit2**](https://github.com/airockchip/rknn-toolkit2/tree/v2.1.0/rknn-toolkit2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknn-toolkit2/packages/rknn_toolkit2-2.1.0+708089d1-cp310-cp310-linux_x86_64.whl

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
      from ultralytics import YOLO
      model = YOLO("yolov8.pt")
      path = model.export(format="rknn")  # Internal method written by airockchip, don't be fooled by the format name
      ```

  1.5 Check the input size when exporting the model. If necessary, change batch_size parameter in ultralytics/cfg/default.yaml to any value.

  2. Convert onnx to rknn.
      ```
      # Clone repo
      git clone https://github.com/airockchip/rknn_model_zoo

      # Go to directory with converter
      cd rknn_model_zoo/examples/yolov8/python

      # Run converter
      python convert.py <path-to-onnx-model>/yolov8n.onnx rk3588 i8 ../model/yolov8n.rknn
      ```

  2.5 If the model has issues or warnings in convertation process, you can change opset version from 12 to 17 or 19, depending on PyTorch version. Currently, RKNN==2.1.0 recommends opset 19.

  3. Save and send it to Orange Pi.

## Install OS

  1. Download image:

      | [Ubuntu (OrangePi 5)](https://drive.google.com/drive/folders/1i5zQOg1GIA4_VNGikFl2nPM0Y2MBw2M0) | [Ubuntu (OrangePi 5B)](https://drive.google.com/drive/folders/1xhP1KeW_hL5Ka4nDuwBa8N40U8BN0AC9) | [Armbian (OrangePi 5/5B)](https://www.armbian.com/orangepi-5/) |
      | :---: | :---: | :---: |

  2. Burn it to SD card.

  3. Plug SD card to Orange Pi.

## Configure OrangePi for running models

  1. Update [**librknnrt.so**](https://github.com/airockchip/rknn-toolkit2/blob/v2.1.0/rknpu2/runtime/Linux/librknn_api/aarch64/).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

      # Move to /usr/lib
      sudo mv ./librknnrt.so /usr/lib
      ```

  2. Install whls for [**rknn-toolkit-lite2**](https://github.com/airockchip/rknn-toolkit2/tree/v2.1.0/rknn-toolkit-lite2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl

      # Install
      pip install rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl
      ```

  3. Install opencv-python and other requirements (if necessary).

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
