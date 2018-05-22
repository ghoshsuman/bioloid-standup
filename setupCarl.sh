#dependencies
#Prima di installare stormpy bisogna installare tutte le dipendenze
#pip è tra le dipendenze ma è stato precedentemente installato

#cmake install
sudo apt-get install -y cmake cmake-curses-gui g++ doxygen

#libcln6 install
sudo apt-get install -y libcln6 libcln-dev libgmp-dev

workig_dir=$(pwd)

#Download and Install Eigen

git clone -b branches/3.3 git@github.com:eigenteam/eigen-git-mirror.git eigen
cd eigen 
mkdir build && cd build/ && cmake ../
sudo make install
# make check

cd $working_dir


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

cd $workig_dir

#Clone pycarl into any suitable location:
git clone https://github.com/moves-rwth/pycarl.git
cd pycarl
python3 setup.py build_ext --carl-dir $carl_dir develop
python -m pytest tests


#After building, you can run the test files by:
#$ 

cd $workig_dir
