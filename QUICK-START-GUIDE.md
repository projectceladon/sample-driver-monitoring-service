# Driver Behaviour

## Pre-requite

* You should install OpenVINO 2022.1 or greater
* You should install or build opencv version 1.46.2 for cpp
* In order to convert model, you should install openvino-dev by PIP.

```shell
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
source <openvino path>/setupvars.sh
bash scripts/download_models.sh
```

## Build Program

```shell
source <openvino path>/setupvars.sh
mkdir build
cd build
source ../scripts/setupenv.sh
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

## Launch Program

### OpenVINO ENV

```shell
source <openvino path>/setupvars.sh
```

### Project ENV

```shell
cd scripts
source setupenv.sh
```

### Run Program

#### Camera Input

```shell
cd ../build/intel64/Release
./driver_behavior -m $face232 -d CPU -m_hp $hp32 -d_hp CPU -dlib_lm
```

#### Recorded Video Input

```shell
cd ../build/intel64/Release
./driver_behavior -m $face232 -d CPU -m_hp $hp32 -d_hp CPU -dlib_lm -i <path/video/input>

# (Example)
./driver_behavior -m $face232 -d CPU -m_hp $hp32 -d_hp CPU -dlib_lm -i ../../../.data/demo03-resized.webm
```
