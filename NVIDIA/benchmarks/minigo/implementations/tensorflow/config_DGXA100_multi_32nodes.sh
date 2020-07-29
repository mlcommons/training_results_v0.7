#
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_multi.sh

# System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:30:00

#
export NUM_ITERATIONS=80
