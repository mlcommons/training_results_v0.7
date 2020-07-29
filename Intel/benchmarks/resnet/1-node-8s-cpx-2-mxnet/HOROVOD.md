<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Multinode-training with Horovod

## Usage

### 1. Install Horovod

1. Install Intel Machine Learning Scaling Library (MLSL) from source.
```bash
git clone https://github.com/intel/MLSL.git mlsl
cd mlsl
make all
make install
```

2. Install Intel MPI 2019 from source
```bash
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15838/l_mpi_2019.5.281.tgz
tar -zxf l_mpi_2019.5.281.tgz
cd l_mpi_2019.5.281
# Intel-MPI will be by default installed into $HOME/intel
./install.sh
```

3. Install Horovod via pip. Remember to export the MXNet PATH by `PYTHONPATH` before installing Horovod. For more details please refer to [Horovod install guide page](https://github.com/horovod/horovod/blob/master/docs/contributors.rst).

```bash
export PYTHONPATH=$(pwd)/resnet/incubator-mxnet/python:$PYTHONPATH
source $(pwd)/resnet/mlsl/_install/intel64/bin/mlslvars.sh thread
source $HOME/intel/impi/2019.5.281/intel64/bin/mpivars.sh release_mt

pip install horovod==0.18.2
```
Then, you will have Horovod and MXNet installed in your server. You can try these below commands to verify if these two have been installed correctly.

```bash
$ python
Python 3.7.4 (default, Aug 13 2019, 20:35:49)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import horovod
>>> import horovod.mxnet as hvd
>>> hvd
<module 'horovod.mxnet' from '/root/miniconda3/envs/multinode/lib/python3.7/site-packages/horovod-0.18.2-py3.7-linux-x86_64.egg/horovod/mxnet/__init__.py'>
```


### 2. Prepare hostfile for distributed training

```bash
hostname -I > hostfile
```

It's worth noting that it is important to make sure that all computing nodes can be access by SSH without password verfication.

### 3. Start training

#### 1. Run with Intel MLSL
We have already provided the scripts to launch multi-node training using MLSL, quick try by running below cmd:

```bash
# Note: the execution order of following two commands cannot be changed since the "I_MPI_ROOT" will be overrode by mlsl.
source $(pwd)/resnet/mlsl/_install/intel64/bin/mlslvars.sh thread
source $HOME/intel/impi/2019.5.281/intel64/bin/mpivars.sh release_mt

# launch training on 4S/8S machine
bash run_multi_cpx.sh resnet50-v1.5_multi_ins_hyve_bf16 hostfile /your/path/to/mxnet/ /path/to/imagenet/dataset/

# or you can just run below command for convenience
bash run_and_time.sh
```

### Note: Post processing on the original results
The post-processing script is provided to make the original log files to pass compliance checker. Just follow the below command:
```python
python post_processing.py --in-file <original_log_file> --out-file "result_0.txt"
```
