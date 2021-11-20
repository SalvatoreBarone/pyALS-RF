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
  ./edginess --ax als --pmml $pmml --output ${pmml%.*} --dataset $test_dataset_file --catalog lut_catalog.db --popsize 50 --iter 111
done
