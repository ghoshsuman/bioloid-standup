#!/bin/sh

working_dir=$(pwd)

#dependencies
sudo apt-get install -y libgmp3-dev libginac-dev automake libglpk-dev libhwloc-dev libcln-dev  
sudo apt-get install -y libz3-dev libxerces-c-dev


#Storm

#The source code of the latest stable release can be downloaded from GitHub. You can either clone the git repository
git clone -b stable https://github.com/moves-rwth/storm.git


cd storm

mkdir build && cd build && cmake ../ -DSTORM_USE_CLN_RF=OFF
make #Compile all of Storm’s binaries including all tests

export STORM_DIR=$(pwd)

cd ${working_dir}

#Stormpy

#Clone stormpy into any suitable location:
git clone https://github.com/moves-rwth/stormpy.git
git checkout tags/1.2.0
cd stormpy
python setup.py build develop
python -m pytest tests

cd ${working_dir}

