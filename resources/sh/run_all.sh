#!/bin/bash
usage() {
        echo "Usage: $0 -v /xilinx/vivado";
        exit 1;
}

while getopts "v:t:" o; do
    case "${o}" in
        v)
            xilinx_vivado=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${xilinx_vivado}" ]; then
    usage
fi

xilinx_vivado=$(realpath $xilinx_vivado)
current_dir=$(pwd); 

for i in $(find . -type d -name 'configuration_*' | sort);
do
  echo $i
  cd $i
  ./run_synth.sh $xilinx_vivado
  cd $current_dir
done
for i in $(find . -type d -name 'configuration_*' | sort);
do 
  echo $i
  cd $i
  ./run_sim.sh $xilinx_vivado
  cd $current_dir
done  
