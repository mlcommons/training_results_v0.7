
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Post processing for MLPerf Minigo logging')
    parser.add_argument("--in-file", type=str, required=True,
                        help="input log file to be processed")
    parser.add_argument("--out-file", type=str, required=True,
                        help="output log file")
    args = parser.parse_args()
    return args

# add epoch num to eval_start/eval_stop metadata
# add evaluation start offset to epoch num, if required.
def correct_eval_startstop(line): 

    words=line.split() 
    epoch=None
    for j in range(len(words)):
        if "value" in words[j]:
            epoch=words[j+1][:-1]
            break
    if int(epoch) < 50: 
       epoch = str(50 + int(epoch)) 
 
    epochstring = ", " + "\"epoch_num\": " + epoch + "}}"
    line = line.replace('}}', epochstring)
    
    return line

# add status to run_stop metadata
def correct_run_stop(line):

    
    statusstring = ", " + "\"status\": " + "\"success\"" + "}}"
    line = line.replace('}}', statusstring)
    
    return line

# add evaluation start offset to epoch num
def correct_eval_epochnum(line):

    words = line.split()
    oldepoch = words[-1][:-2]    
    if int(oldepoch) < 50:
        newepoch=50+int(oldepoch) 
        line = line.replace(oldepoch+"}}",str(newepoch)+"}}")       

    return line

def main():

  args = parse_args()
  in_file  = open(args.in_file, "r")
  out_file = open(args.out_file, "w")


  n_evalsamples=0
  n_mingames=0
  
  for line in in_file.readlines():

     if 'eval_start' in line or 'eval_stop' in line:
        line = correct_eval_startstop(line)
     if 'run_stop' in line:
        line = correct_run_stop(line)
     if 'eval_accuracy' in line:
        line = correct_eval_epochnum(line)
    
     if 'eval_samples' in line: 
        if n_evalsamples==0: 
           n_evalsamples =1 
        else:
           line = line.replace('eval_samples', 'dummy') 
     
     if 'min_selfplay_games_per_generation' in line:   
        if n_mingames==0:
           n_mingames=1
        else:
           line = line.replace('min_selfplay_games_per_generation', 'dummy')
     out_file.write(line) 
  
  in_file.close()
  out_file.close()  

if __name__ == "__main__": main()  


