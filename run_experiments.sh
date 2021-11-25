#!/bin/bash
usage() {
        echo "Usage: $0 -d /pmml_directory -t test_dataset_file";
        exit 1;
}

while getopts "d:t:" o; do
    case "${o}" in
        d)
            directory=${OPTARG}
            ;;
        t)
            test_dataset_file=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${directory}" ] || [ -z "${test_dataset_file}" ] ; then
    usage
fi

directory=$(realpath $directory)
test_dataset_file=$(realpath $test_dataset_file)

for pmml in $(find $directory -name '*.pmml' | sort);
do
  rm -rf ${pmml%.*}
  mkdir -p ${pmml%.*}
  ./edginess --ax als --catalog lut_catalog.db --lut 6 --timeout 120000 --pmml $pmml --output ${pmml%.*} --dataset $test_dataset_file --popsize 50 --iter 51 --pcross 0.8 --etac 100 --pmut 0.10 --etam 10 --emax 5 | tee ${pmml%.*}/log.txt
done
