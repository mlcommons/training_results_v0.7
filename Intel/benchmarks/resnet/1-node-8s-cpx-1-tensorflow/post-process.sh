#!/bin/bash

# generate submission details 
python generate_submission_details.py > head

# change some opt_variables to lars_opt_variables
sed -i 's/opt_base_learning_rate/lars_opt_base_learning_rate/g' $1
sed -i 's/opt_learning_rate_warmup_epochs/lars_opt_learning_rate_warmup_epochs/g' $1

# putting epoch_num and epoch_count and first_epoch_num into the fields
awk -f post-process.awk $1 > temp_1.txt

# combine pieces
cat temp_1.txt >> head
mv head temp_1.txt

#delete one duplicate run_stop
cat temp_1.txt | head -n -2 > tmp
tail -n 1 temp_1.txt > tmp2
mv tmp temp_1.txt
cat tmp2 >> temp_1.txt

mv temp_1.txt $1 
rm tmp2

