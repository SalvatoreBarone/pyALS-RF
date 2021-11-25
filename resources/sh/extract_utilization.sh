#!/bin/bash
echo "Configuration | Total LUTs | Logic LUTs | FFs"
for i in $(find . -name 'report_utilization.txt' | sort); 
do 
  total_luts=$(grep -e " classifier " $i | sed "s/ \+//g" | cut -d '|' -f4); 
  logic_luts=$(grep -e " classifier " $i | sed "s/ \+//g" | cut -d '|' -f5); 
  total_ffls=$(grep -e " classifier " $i | sed "s/ \+//g" | cut -d '|' -f8); 
  echo $i $total_luts $logic_luts $total_ffls;
done     

