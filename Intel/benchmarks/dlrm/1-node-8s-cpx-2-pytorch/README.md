# DLRM MLPerf Training v0.7 Intel Submission

## HW and SW requirements
### 1. HW requirements
| HW | configuration |
| -: | :- |
| CPU | CPX-6 @ 8 sockets/Node |
| DDR | 192G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T |

### 2. SW requirements
| SW |configuration  |
|--|--|
| GCC | GCC 8.3  |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
   wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
   chmod +x anaconda3.sh
   ./anaconda3.sh -b -p ~/anaconda3
   ~/anaconda3/bin/conda create -n dlrm python=3.7

   export PATH=~/anaconda3/bin:$PATH
   source ./anaconda3/bin/activate dlrm
```
### 2. Install dependency packages
```
   pip install sklearn onnx tqdm lark-parser
   pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging

   conda config --append channels intel
   conda install ninja pyyaml setuptools cmake cffi typing
   conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps
   conda install -c conda-forge gperftools
```
### 3. Clone source code and Install
(1) Install PyTorch and Intel Extension for PyTorch
```
   # clone PyTorch
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
   git submodule sync && git submodule update --init --recursive

   # clone Intel Extension for PyTorch
   git clone https://github.com/intel/intel-extension-for-pytorch.git
   cd intel-extension-for-pytorch && git checkout tags/v0.2 -b v0.2
   git submodule update --init --recursive

   # install PyTorch
   cd {path/to/pytorch}
   cp {path/to/intel-pytorch-extension}/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
   patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch
   python setup.py install

   # install Intel Extension for PyTorch
   cd {path/to/intel-pytorch-extension}
   python setup.py install
```
(2) Install oneCCL
```
   git clone https://github.com/oneapi-src/oneCCL.git
   cd oneCCL && git checkout 2021.1-beta07-1
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
   make install -j
```
(3) Install Torch CCL
```
   git clone https://github.com/intel/torch-ccl.git
   cd torch-ccl && git checkout 2021.1-beta07-1
   source ~/.local/env/setvars.sh
   python setup.py install
```
(4) Install DLRM
```
   git clone https://github.com/facebookresearch/dlrm.git
   cd dlrm && git checkout mlperf
   cp {path/to/intel-pytorch-extension}/torch_patches/models/0001-enable-dlrm-distributed-training-for-cpu.patch .
   patch -p1 < 0001-enable-dlrm-distributed-training-for-cpu.patch
```
## 4. Run command
(1) 32K global BS with 8 ranks (1 Hyve/CDS Node).
```
   # Clean resources
   echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
   echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
   echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
   echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
   echo 1 > /proc/sys/vm/compact_memory; sleep 1
   echo 3 > /proc/sys/vm/drop_caches; sleep 1
```
```
   DATA_PATH={dataset path}
   seed_num=$(date +%s)
   export KMP_BLOCKTIME=1
   export KMP_AFFINITY="granularity=fine,compact,1,0"
   export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
   export CCL_WORKER_COUNT=8
   export CCL_WORKER_AFFINITY="0,1,2,3,4,5,6,7,28,29,30,31,32,33,34,35,56,57,58,59,60,61,62,63,84,85,86,87,88,89,90,91,112,113,114,115,116,117,118,119,140,141,142,143,144,145,146,147,168,169,170,171,172,173,174,175,196,197,198,199,200,201,202,203"
   export CCL_ATL_TRANSPORT=ofi
   export MASTER_ADDR="127.0.0.1"

   Config="mpiexec.hydra -np 8 -ppn 8 -l -genv I_MPI_PIN_DOMAIN=[0x0000000000000000000000000000000000000000000000000FFFFF00,0x000000000000000000000000000000000000000000FFFFF000000000,0x00000000000000000000000000000000000FFFFF0000000000000000,0x0000000000000000000000000000FFFFF00000000000000000000000,0x000000000000000000000FFFFF000000000000000000000000000000,0x00000000000000FFFFF0000000000000000000000000000000000000,0x0000000FFFFF00000000000000000000000000000000000000000000,0xFFFFF000000000000000000000000000000000000000000000000000] -genv OMP_NUM_THREADS=20"
   $Config python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000  --data-generation=dataset --data-set=terabyte --raw-data-file=$DATA_PATH/day --processed-data-file=$DATA_PATH/terabyte_processed.npz --loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex --bf16  --dist-backend=ccl --learning-rate=18.0 --mini-batch-size=32768 --print-freq=128 --print-time --test-freq=6400 --test-mini-batch-size=65536 --lr-num-warmup-steps=8000 --lr-decay-start-step=70000 --lr-num-decay-steps=30000 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=$seed_num $dlrm_extra_option 2>&1 | tee run_terabyte_mlperf_32k_8_sockets.log
```
