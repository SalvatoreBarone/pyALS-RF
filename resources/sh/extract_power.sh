#!/bin/bash
echo "Configuration | Total Power "
for i in $(find . -name 'report_power.txt' | sort); 
do 
  total_power=$(grep -e "^| classifier " $i | sed "s/ \+//g" | cut -d '|' -f3); 
  echo $i $total_power;
done     


