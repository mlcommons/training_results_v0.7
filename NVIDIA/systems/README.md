# DGXOS - HPC Configuration Instructions

Install instructions for DGX OS HPC.

<br>

---
**WARNING:** Once these changes have been made, the changes can only be removed with a system reimage. Uninstalling/removing the package with `apt-get` will not revert these changes. Ensure your system is properly backed up before continuing.

**WARNING:** By making these changes, all the Spectre/Meltdown/MDS and other mitigations are being disabled. This configuration is only expected to run in a trusted environment.

---
<br>

## Requirement
DGX-1, DGX-2, DGX-2H, or DGX A100
  

## Prerequisites
### Base OS

#### DGX-1, DGX-2, DGX-2H

These instructions require that DGX OS 4.4.0 is installed. To check which OS version is currently installed, run the following command:  

`grep VERSION /etc/dgx-release`  
```bash
DGX_SWBUILD_VERSION="4.4.0"
```

Before beginning the installation, please ensure you are running Base OS 4.4.0. If not, follow the [NVIDIA DGX OS SERVER VERSION 4.4.0 Release Notes and Update Guide](https://docs.nvidia.com/dgx/pdf/DGX-OS-server-4.4-relnotes-update-guide.pdf) before continuing.

#### DGX A100

These instructions require that DGX OS 4.99.8 is installed. To check which OS version is currently installed, run the following command:  

`grep VERSION /etc/dgx-release`  
```bash
DGX_SWBUILD_VERSION="4.99.8"
```

Before beginning the installation, please ensure you are running Base OS 4.99.8. If not, follow the [NVIDIA DGX OS SERVER VERSION 4.99.8 Release Notes and Update Guide](https://docs.nvidia.com/dgx/pdf/DGX-OS-server-4.99-relnotes-update-guide.pdf) before continuing.


### Kernel and Security Updates
The latest kernel and other security patches must be installed. Run the following commands to ensure your machine is up to date: 

`sudo apt-get update`
`sudo apt-get dist-upgrade`
`sudo apt-get autoremove --purge`

Depending on the DGX configuration, it's possible the command `sudo apt-get dist-upgrade` will
display one or more user prompts asking whether to keep a local config file 
or take the package maintainer's version. 
Accept the default option when prompted, or run with `-y` to do this automatically.

## Installation

1. Set up kernel command line.

    For DGX A100, add the file `/etc/default/grub.d/hpc.cfg` with following contents:

    ```
    GRUB_CMDLINE_LINUX="${GRUB_CMDLINE_LINUX} nvme-core.multipath=n cgroup_enable=memory swapaccount=1 mitigations=off "
    ```

    For DGX-1, DGX-2, or DGX-2H, add the file `/etc/default/grub.d/hpc.cfg` with following contents instead:

    ```
    GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} console=tty0 console=ttyS1,115200 "
    GRUB_CMDLINE_LINUX="${GRUB_CMDLINE_LINUX} nvme-core.multipath=n cgroup_enable=memory swapaccount=1 noibpb pti=off mitigations=off transparent_hugepage=madvise module_blacklist=nouveau "
    ```

2. Update Grub and initramfs.

   `sudo update-grub`

   `sudo update-initramfs -u`

2. Add the file `/etc/security/limits.d/zz-dgxos-hpc-limits.conf` with the following contents:

```
* soft memlock unlimited
* hard memlock unlimited
* hard memlock unlimited
* soft stack unlimited
* hard stack unlimited
* soft nofile 65536
* hard nofile 65536
```

3. Add the file `/etc/sysctl.d/60-dgxos-hpc-sysctl.conf` with the following contents:

```
vm.dirty_ratio = 5
vm.min_free_kbytes = 1048576
vm.dirty_writeback_centisecs = 100
vm.dirty_expire_centisecs = 100
vm.max_map_count = 1048576
kernel.yama.ptrace_scope = 0
kernel.core_pattern = /dev/shm/%e_%p.core
```

4. Purge NVSM

   `sudo apt-get purge dgx-limits mongodb-clients mongodb-server mongodb-server-core nvsm-apis nvsm-cli nvsm-dshm nvsm`

5. Purge fail2ban

   `sudo apt-get purge fail2ban`

6. Remove any leftover dependencies

   `sudo apt-get autoremove --purge`

7. Restart the server:

   `sudo reboot`

8. Verify that CPU mitigations that impact performance are disabled:

```bash
$ grep -o 'mitigations=off' /proc/cmdline
mitigations=off
```
