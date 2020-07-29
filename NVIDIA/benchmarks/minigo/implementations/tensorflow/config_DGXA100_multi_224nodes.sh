#
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_multi.sh

# System run parms
export DGXNNODES=224
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:45:00
