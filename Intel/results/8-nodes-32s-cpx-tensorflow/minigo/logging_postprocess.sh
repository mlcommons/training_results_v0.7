log=$1
std_out=$2
log2=$3

if [[ -e $log ]] && [[ -e $std_out ]] ;then

echo $log,$std_out

# Get the time to train by ms
ttt=$(grep "beat target after" $std_out |awk -F' ' '{ print $6 }'|sed 's/s//g')
echo "beat taget after $ttt s"

start=$(grep "run_start" $log |awk -F',' '{ print $2 }' |awk -F':' '{ print $2 }')
echo "run_start at $start"

stop=`bc <<< "$start + 1000 * $ttt"`
echo "run_stop at $stop"
sed "s#.*run_stop.*succuss.*#:::MLLOG {\"namespace\": \"worker1\", \"time_ms\": $stop, \"event_type\": \"INTERVAL_END\", \"key\": \"run_stop\", \"value\": \"succuss\", \"metadata\": {\"file\": \"ml_perf\/eval_models.py\", \"lineno\": 171, \"status\": \"success\"}}#g" $log >$log2

fi
