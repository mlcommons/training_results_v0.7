import collections
import os
import subprocess
import numpy as np

from mlperf_logging.mllog import constants as mlperf_constants
from mlperf_logging import mllog

class MPIWrapper(object):
    def __init__(self):
        self.comm = None
        self.MPI = None

    def get_comm(self):
        if self.comm is None:
            import mpi4py
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI = MPI

        return self.comm

    def barrier(self):
        c = self.get_comm()
        # NOTE: MPI_Barrier is *not* working reliably at scale. Using MPI_Allreduce instead.
        #c.Barrier() 
        val = np.ones(1, dtype=np.int32)
        result = np.zeros(1, dtype=np.int32)
        c.Allreduce(val, result)

    def allreduce(self, x):
        c = self.get_comm()
        rank = c.Get_rank()
        val = np.array(x, dtype=np.int32)
        result = np.zeros_like(val, dtype=np.int32)
        c.Allreduce([val, self.MPI.INT], [result, self.MPI.INT]) #, op=self.MPI.SUM)
        return result

    def rank(self):
        c = self.get_comm()
        return c.Get_rank()

mpiwrapper=MPIWrapper()

def all_reduce(v):
    return mpiwrapper.allreduce(v)

def mx_resnet_print(key, val=None, metadata=None, deferred=False, stack_offset=1,
                    sync=False, uniq=True):
    rank = mpiwrapper.rank()
    if sync:
        mpiwrapper.barrier()

    if (uniq and rank == 0) or (not uniq):
        mllogger = mllog.get_mllogger()
        if key == mlperf_constants.RUN_START:
            mllogger.start(key=key, value=val, metadata=metadata)
        elif key== mlperf_constants.RUN_STOP:
            mllogger.end(key=key, value=val, metadata=metadata)
        else:
            mllogger.event(key=key, value=val, metadata=metadata)

    if sync:
        mpiwrapper.barrier()

    return

def mlperf_submission_log(benchmark):

    framework = "MXNet NVIDIA Release {}".format(os.environ["NVIDIA_MXNET_VERSION"]);

    def query(command):
        result = subprocess.check_output(
            command, shell=True,
            stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        return result

    def get_sys_storage_type():
        dev = query('lsblk -e 11 -ndio KNAME | head -1')
        if dev.startswith('sd'):
            transport = 'SATA'
        elif dev.startswith('hd'):
            transport = 'IDE'
        elif dev.startswith('nvme'):
            transport = 'NVMe'
        else:
            transport = '<unknown bus>'

        disk_type = 'SSD'
        sys_storage_type = '{} {}'.format(transport, disk_type)
        return sys_storage_type

    def get_interconnect():
        dev = query('ibstat -l | head -1')
        link_layer = query('ibstatus {} | grep "link_layer" | cut -f 2- -d" "'.format(dev))
        rate = query('ibstatus {} | grep "rate" | cut -f 2- -d" "'.format(dev))
        interconnect = '{} {}'.format(link_layer, rate)
        return interconnect

    def get_sys_mem_size():
        sys_mem_size = query(
            "grep 'MemTotal' '/proc/meminfo' | awk '{ print $2 }'"
            )
        sys_mem_size = '{} GB'.format(int(sys_mem_size) // (1024 * 1024))
        return sys_mem_size

    def get_sys_storage_size():
        sizes = query(
            'lsblk -e 11 -dno SIZE | sed \'s/ //g\''
            ).split()
        sizes_counter = collections.Counter(sizes)
        sys_storage_size = ' + '.join(['{}x {}'.format(val, key) for key, val in sizes_counter.items()])
        return sys_storage_size

    def get_cpu_interconnect(cpu_model):
        if cpu_model == '85':
            # Skylake-X
            cpu_interconnect = 'UPI'
        else:
            cpu_interconnect = 'QPI'
        return cpu_interconnect

    gcc_version = query(
        'gcc --version |head -n1'
        )

    os_version = query(
        'cat /etc/lsb-release |grep DISTRIB_RELEASE |cut -f 2 -d "="',
        )
    os_name = query(
        'cat /etc/lsb-release |grep DISTRIB_ID |cut -f 2 -d "="',
        )

    cpu_model = query(
        'lscpu |grep "Model:"|cut -f2 -d:'
        )
    cpu_model_name = query(
        'lscpu |grep "Model name:"|cut -f2 -d:'
        )
    cpu_numa_nodes = query(
        'lscpu |grep "NUMA node(s):"|cut -f2 -d:'
        )
    cpu_cores_per_socket = query(
        'lscpu |grep "Core(s) per socket:"|cut -f2 -d:'
        )
    cpu_threads_per_core = query(
        'lscpu |grep "Thread(s) per core:"|cut -f2 -d:'
        )

    gpu_model_name = query(
        'nvidia-smi -i 0 --query-gpu=name --format=csv,noheader,nounits'
        )
    gpu_count = query(
        'nvidia-smi -i 0 --query-gpu=count --format=csv,noheader,nounits'
        )

    sys_storage_size = get_sys_storage_size()

    hardware = query(
        'cat /sys/devices/virtual/dmi/id/product_name'
        )

    network_card = query(
        'lspci | grep Infiniband | grep Mellanox | cut -f 4- -d" " | sort -u'
        )
    num_network_cards = query(
        'lspci | grep Infiniband | grep Mellanox | wc -l'
        )
    mofed_version = query(
        'cat /sys/module/mlx5_core/version'
        )
    interconnect = get_interconnect()

    cpu = '{}x {}'.format(cpu_numa_nodes, cpu_model_name)
    num_cores = '{}'.format(int(cpu_numa_nodes) * int(cpu_cores_per_socket))
    num_vcores = '{}'.format(int(num_cores) * int(cpu_threads_per_core))
    cpu_interconnect = get_cpu_interconnect(cpu_model)

    sys_storage_type = get_sys_storage_type()
    sys_mem_size = get_sys_mem_size()

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)

    nodes = {
        'num_nodes': num_nodes,
        'cpu': cpu,
        'num_cores': num_cores,
        'num_vcpus': num_vcores,
        'accelerator': gpu_model_name,
        'num_accelerators': gpu_count,
        'sys_mem_size': sys_mem_size,
        'sys_storage_type': sys_storage_type,
        'sys_storage_size': sys_storage_size,
        'cpu_accel_interconnect': cpu_interconnect,
        'network_card': network_card,
        'num_network_cards': num_network_cards,
        'notes': '',
        }

    libraries = {
        'container_base': '{}-{}'.format(os_name, os_version),
        'openmpi_version': os.environ['OPENMPI_VERSION'],
        'mofed_version': mofed_version,
        'cuda_version': os.environ['CUDA_VERSION'],
        'cuda_driver_version': os.environ['CUDA_DRIVER_VERSION'],
        'nccl_version': os.environ['NCCL_VERSION'],
        'cudnn_version': os.environ['CUDNN_VERSION'],
        'cublas_version': os.environ['CUBLAS_VERSION'],
        'trt_version': os.environ['TRT_VERSION'],
        'dali_version': os.environ['DALI_VERSION'],
        }

    entry = {
        'hardware': hardware,
        'framework': framework,
        'power': 'N/A',
        'notes': 'N/A',
        'interconnect': interconnect,
        'os': os.environ.get('MLPERF_HOST_OS', '').replace('_',' '),
        'libraries': str(libraries),
        'compilers': gcc_version,
        'nodes': str(nodes),
        }

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_BENCHMARK,
        val=benchmark,
        )

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_ORG,
        val='Fujitsu')

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_DIVISION,
        val='closed')

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_STATUS,
        val='onprem')

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_PLATFORM,
        val='1xGX2570M5')

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_POC_NAME,
        val=('Akihiko Kasagi')

    mx_resnet_print(
        key=mlperf_constants.SUBMISSION_POC_EMAIL,
        val='kasagi.akihiko@fujitsu.com')


def resnet_max_pool_log(input_shape, stride):
    downsample = 2 if stride == 2 else 1
    output_shape = (input_shape[0], 
                    int(input_shape[1]/downsample), 
                    int(input_shape[2]/downsample))

    return output_shape


def resnet_begin_block_log(input_shape, block_type):
    return input_shape


def resnet_end_block_log(input_shape):
    return input_shape


def resnet_projection_log(input_shape, output_shape):
    return output_shape


def resnet_conv2d_log(input_shape, stride, out_channels, initializer, bias):
    downsample = 2 if (stride == 2 or stride == (2, 2)) else 1
    output_shape = (out_channels, 
                    int(input_shape[1]/downsample), 
                    int(input_shape[2]/downsample))

    return output_shape


def resnet_relu_log(input_shape):
    return input_shape


def resnet_dense_log(input_shape, out_features):
    shape = (out_features)
    return shape


def resnet_batchnorm_log(shape, momentum, eps, center=True, scale=True, training=True):
    return shape
