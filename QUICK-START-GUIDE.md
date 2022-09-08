# Driver Behaviour

## Pre-requite

* You should install OpenVINO 2022.1 by APT.
* In order to convert model, you should install openvino-dev by PIP.

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install wheel
python -m pip install openvino-dev
```

* Necessary packages to build

```shell
sudo apt-get install libboost-dev libboost-log-dev libao-dev libsndfile1-dev libx11-dev libopenblas-dev liblapack-dev libgflags-dev libjpeg9-dev libopenblas-dev libblas-dev
```

## Download models

```shell
source /opt/intel/openvino_2022/setupvars.sh
source .venv/bin/activate
bash scripts/download_models.sh
```

## Build Program

```shell
source /opt/intel/openvino_2022/setupvars.sh
mkdir build
cd build
source ../scripts/setupenv.sh
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

## Launch Program

### OpenVINO ENV

```shell
source /opt/intel/openvino_2022/setupvars.sh
```

### Project ENV

```shell
cd scripts
source setupenv.sh
```

### Register Face

* Put your image to `drivers` folder. And file name should be `<name>.N.jpg`

```shell
cd scripts/
python3 create_list.py ../drivers/
```

### Run Program

```shell
cd ../build/intel64/Release
./driver_behavior -m $face232 -d CPU -m_hp $hp32 -d_hp CPU -dlib_lm -d_recognition -fg ../../../scripts/faces_gallery.json
```
