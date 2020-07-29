#
source $(dirname ${BASH_SOURCE[0]})/config_DGX2_multi.sh

## System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:30:00
