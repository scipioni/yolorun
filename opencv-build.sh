#!/bin/bash

# https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.6.0_Jetson.sh

set -x
RELEASE=$1
RELEASE=${RELEASE:=4.8.0}
PYTHON=$2
PYTHON=${PYTHON:=3.10}


# xavier CUDA_ARCH_BIN=7.2
#CUDA="-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_GENERATION=Auto"

rm -fR opencv-$RELEASE

if [ ! -d opencv-$RELEASE ]; then
	[ -f opencv-$RELEASE.tar.gz ] || curl -L https://github.com/opencv/opencv/archive/${RELEASE}.tar.gz -o opencv-${RELEASE}.tar.gz
	tar zxf opencv-$RELEASE.tar.gz
fi

if [ ! -d opencv_contrib-$RELEASE ]; then
	curl -L https://github.com/opencv/opencv_contrib/archive/${RELEASE}.tar.gz -o opencv_contrib-${RELEASE}.tar.gz
	tar zxf opencv_contrib-${RELEASE}.tar.gz
fi

rm -fR opencv-$RELEASE/release
mkdir -p opencv-$RELEASE/release
cd opencv-$RELEASE/release


JOBS=$(getconf _NPROCESSORS_ONLN)
JOBS=$(($JOBS - 1)) 

# with CUDA 10 disable cudacodec
# -D BUILD_opencv_cudacodec=OFF
#
#-D CUDA_ARCH_BIN="7.2,8.7" \
#-D CUDA_ARCH_PTX="" \

#-D CUDA_ARCH_PTX="" \
#-D CUDA_ARCH_BIN="7.2,8.7" \
# 2023
cmake \
	-D BUILD_LIST=core,improc,videoio,dnn,python3,cudev,dnn_objdetect,highgui,video,calib3d,gapi \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D CUDA_GENERATION=Auto \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${RELEASE}/modules \
	-D WITH_GSTREAMER=ON \
	-D WITH_LIBV4L=ON \
	-D PYTHON_EXECUTABLE=$(which $PYTHON) \
	-D BUILD_opencv_python3=ON \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/build/release \
	-D OPENCV_PYTHON3_INSTALL_PATH=$($PYTHON -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
	..


#-D CMAKE_INSTALL_PREFIX=$($PYTHON -c "import sys; print(sys.prefix)") \
make -j${JOBS} install/strip

