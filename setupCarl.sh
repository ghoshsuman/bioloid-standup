#!/bin/sh



#cmake install
sudo apt-get install -y cmake cmake-curses-gui g++ doxygen

#libcln6 install
sudo apt-get install -y libcln6 libcln-dev libgmp-dev

working_dir=$(pwd)

#Download and Install Eigen

git clone -b branches/3.3 git@github.com:eigenteam/eigen-git-mirror.git eigen
cd eigen 
mkdir build && cd build/ && cmake ../
sudo make install
# make check

cd ${working_dir}


#Download and Install Carl
git clone https://github.com/smtrat/carl.git
cd carl/
mkdir build && cd build/ && cmake ../


#ATTENZIONE! Questa operazione potrebbe richiedere parecchio tempo
#This builds the shared library build/libcarl.so
make lib_carl
# Build carl (with tests and documentation). 
make
make test doc

carl_dir=$(pwd)

#Pycarl
cd ${working_dir}

git clone https://github.com/moves-rwth/pycarl.git
cd pycarl
python setup.py build develop
python -m pytest tests


cd ${working_dir}
