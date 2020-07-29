mkdir -p $HOME/dlrm

# install anaconda
cd $HOME/dlrm
wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
./anaconda3.sh -b -p ~/anaconda3
~/anaconda3/bin/conda create -n dlrm python=3.7

export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/bin/activate dlrm

# install depedency packages
pip install sklearn onnx tqdm lark-parser
pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging

conda config --append channels intel
conda install ninja pyyaml setuptools cmake cffi typing
conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps
conda install -c conda-forge gperftools

# clone PyTorch
cd $HOME/dlrm
git clone https://github.com/pytorch/pytorch.git
cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
git submodule sync && git submodule update --init --recursive

# clone Intel Extension for PyTorch
cd $HOME/dlrm
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch && git checkout tags/v0.2 -b v0.2
git submodule update --init --recursive

# install PyTorch
cd $HOME/dlrm/pytorch
cp $HOME/dlrm/intel-extension-for-pytorch/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch
python setup.py install

# install Intel Extension for PyTorch
cd $HOME/dlrm/intel-extension-for-pytorch
python setup.py install

# install oneCCL
cd $HOME/dlrm
git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL && git checkout 2021.1-beta07-1
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
make install -j

# install torch-ccl
cd $HOME/dlrm
git clone https://github.com/intel/torch-ccl.git
cd torch-ccl && git checkout 2021.1-beta07-1
source ~/.local/env/setvars.sh
python setup.py install

# install dlrm
cd $HOME/dlrm
git clone https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout mlperf
cp $HOME/dlrm/intel-extension-for-pytorch/torch_patches/models/0001-enable-dlrm-distributed-training-for-cpu.patch .
patch -p1 < 0001-enable-dlrm-distributed-training-for-cpu.patch
