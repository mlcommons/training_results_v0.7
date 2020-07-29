## Postprocessing of log files

Folder raw_logs: contains the mlperf log file (train_run*.log) and std out from model evaluation (eval_run*.stdout)  

postprocessing.py: corrects errors in logging to pass compliance checks \
logging_postprocess.sh: updates the time stamp on run_stop based on std out from model evaluation \
process_loop.sh: do postprocessing for all log files in raw_logs 


To generate the result files

```
cd /path/to/Intel/results/8-nodes-32s-cpx-tensorflow/minigo
./process_loop.sh 

```