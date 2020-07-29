#!/bin/bash

# downlode date
for i in $(seq 1 23) 
do  
aria2c -x 16  http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz
done



