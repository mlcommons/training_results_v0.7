################################################################################
##
##  plot_nvprof  
##  Input is a nvprof profile from MXNET
## output is a excel / csv / json object
################################################################################
import getopt
import sys
import sqlite3
import re
import os
import pandas            as pd
import numpy             as np
import subprocess
import binascii
from  yaml import load, dump
from  yaml import Loader, Dumper
import json
from datetime import datetime
################################################################################
## Algorithm to Link GPU events to Layer names
##
##  For each GPU event in CONCURRENT_KERNELS
##  Use the correlation ID to map the GPU event to the runtime cuda event (function call)
##  Now for this Runtime event - record the start time and end time
##  Then Go to the markers - find all markers whose start times are > runtime event
##  start and whose end time is > runtime event end
##   There should be 1 Marker that meets this criteria
################################################################################
################################################################################
## Global Variables
################################################################################
## GetOpt set up
options      = 'h:o:i'     ## Help message - string of possible 1 char options, ':' after option means it takes an arg
long_options      = ['in_files=', 'out_file=', 'debug', 'help', 'show_tables', 'heartbeat=',\
                     'prune=','no_average', 'graph=', 'framework='] ## List of long form options
db_file_list      = []    ## Input file seql DB
pivot_tbl         = None    ## Output pivot table
excel_file_name   = None
excel_writer      = None   ## Pandas excel file writing handle
string_hash       = {}     ## Hash table - maps string ID to name
kernel_hash       = {}     ## Hash table - stores demangled kernel names
time_base         = -1     ## Starting time stamp of experiment
Debug             = False  ## True for print debugging
ComputeAverage    = True   ## False for disabling the average of per layer times
max_int32         = 1 << 32 ## Max 32 bit val
MAX_EXCEL_SHEET_LEN = 31
marker_tbl_index  = {}
marker_tbl_size   = 0
# Used to prune away entries
prune_enable = False
prune_marker = ""
print_all_tbls    = False
HeartBeat         = 0
FwType            = None
graph_input_file  = None
graph_info_map    = {}
Supported_Fw      = ["FW_TENSOR_FLOW", "FW_TENSORRT", "FW_PYTORCH","FW_MXNET", "FW_CAFFE2"]
################################################################################
## Function definitions
################################################################################
################################################################################
##
##  usage()
##
##    print a help message then exit 0
##
################################################################################
def usage ():
    "Print a help message then exit"
    print("Usage: plot_nvprof [-h] --in_files nvp_sqlite_file,nvp_file1,nvp_file2 -out_file\
        output_file_name [--show_tables] [--debug] [--framework] [--prune]")
    sys.exit (0)

################################################################################
##
## parse_cmd_line()
##
##   Uses getopt to parse cmd line options from user
##
################################################################################
def parse_cmd_line () :
    "Uses getopt to parse cmd line options from user"
    ## Exception handling 
    try:
        opts, extra_args = getopt.gnu_getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError as err :
        print ("Exception caught :  {0}".format(err))    ## Didn't specify type of err in format specifier
        sys.exit(1)

    ## Mark this as global scope because other functions need this value
    global db_file_list
    global Debug
    global pivot_tbl
    global excel_file_name
    global excel_writer
    global print_all_tbls
    global HeartBeat
    global ComputeAverage
    global graph_input_file
    global FwType
    global prune_enable
    global prune_marker

    ## Walk list of cmd line options - opts is a pair<string,string>
    for opt, arg in opts:
        if (opt == "-i" or opt == "--in_files"):
            db_file_list = re.split(',', arg)
            print ("Reading in_file {0}".format(arg))
        elif (opt == "-o" or opt == "--out_file"):
            print ("Writing out file {0:s}".format(arg))
            pivot_tbl = arg
            excel_file_name = re.sub(r'.txt', r'.xlsx', pivot_tbl) 
            excel_writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
        elif (opt == "-h" or opt == "--help"):
            usage();
        elif (opt == "-d" or opt == "--debug") :
            print("Enabling Debug print messages")
            Debug = True
        elif (opt == "-s" or opt == "--show_tables") :
            Debug = True
            print_all_tbls = True
        elif (opt == "--graph"):
            graph_input_file = arg 
        elif (opt == "--framework"):
            FwType = arg 
            if FwType not in Supported_Fw:
                raise Exception ("Framework option {} not supported, must be one of {}", FwType, Supported_Fw)
        elif (opt == "-b" or opt == "--heartbeat") :
            HeartBeat = int(arg)
        elif (opt == "-a" or opt == "--no_average") :
            ComputeAverage = False
        elif (opt=='--prune'):
            prune_enable = True
            prune_marker = arg
    return
################################################################################
##
## open_output_files
################################################################################
def open_ouput_file() :
    """
    Check to see if output file specified on cmd line, else use stdout
    """
    global pivot_tbl
    ## Open the output file (pivot_table)
    if(pivot_tbl is None) :
        file_des = sys.stdout
    else :
        file_des = open(pivot_tbl, "w")
    return file_des

################################################################################
##
## reset_global_vars
################################################################################
def reset_global_vars() :
    global string_hash
    time_base = -1
    del string_hash
    string_hash = {}

    return

################################################################################
##
## read_db_file
##
##   Read in database
################################################################################
def read_db_file (db_name=None, output_fd=None):
    global print_all_tbls
    "Read in the DB file and extract relevant tables"
    if db_name is None or output_fd is None:
        print("Error read_db_file: No db file specified - exiting. ")
        sys.exit(1)
    if not os.path.isfile(db_name):
        print("Error read_db_file: file {} not found".format(db_name))
        sys.exit(1)

    print ("Reading DB file {0}".format(db_name))
    connection = sqlite3.connect(db_name)
    cur        = connection.cursor();
    cur.execute("select name from sqlite_master where type='table'")
    #dump_cur(cur)
    all_tbls = get_tbl_names(cur)
    print("All tables {}".format(all_tbls))
    remaining_tbls = []

    ## Read in StringTable and DRIVER first to extract global info used by other table processing
    for tbl in all_tbls:
        update_list = 1
        if not print_all_tbls:
            #for tbl_type in ['DRIVER', 'StringTable', 'RUNTIME', 'MARKER$']:
            #pattern = re.compile(tbl_type)
            #pattern = re.compile(r"(DRIVER|StringTable|RUNTIME|MARKER$)")
            pattern = re.compile(r"(DRIVER|StringTable)")
            res     = re.search(pattern, tbl)
            if res is not None: 
                tbl_type = res.group(1)
                print("Processing table {}".format(tbl_type))
                process_tbl(tbl, cur, tbl_type)
                update_list = 0
        if update_list :
            remaining_tbls.append(tbl) 

    # Walk the remaining list of tables
    if(Debug or print_all_tbls) :
        for tbl in remaining_tbls:
            print ("Tbl {0:s}".format(tbl))
            #process_runtime_tbl(tbl, cur)
            tbl_str = re.sub(r".*_KIND_", "", tbl)
            print ("tbl str {0} from table {1}". format(tbl_str, tbl))
            print("Processing table {}".format(tbl_str))
            process_tbl(tbl, cur, tbl_str)

    if (print_all_tbls) :
        print ("Option --show_tables set - exiting after printing tables")
        sys.exit(0)

    ## Layer names (CPU Runtime) to the kernels (GPU) that they launch
    panda_frame = link_kernel_to_dl_layer(cur, all_tbls, db_name, output_fd)
    if prune_enable :
        panda_frame = prune(panda_frame)
    # Mxnet markers today don't have wgrad and drad in the names, however the kernels do
    if FwType=='FW_MXNET':
        panda_frame = mxnet_kernel_to_marker(panda_frame)
    
    connection.close()

    ## Clear globals that are set up on each pass of the db file
    reset_global_vars()

    return panda_frame

################################################################################
## kernel_to_phase()
################################################################################
def kernel_to_phase(row):
    ph = row['s_phase']
    if ph == 'fprop':
        return ph
    else:
        k = row['s_kernel'].lower()
        if k.find('wgrad')>=0:
            return 'wgrad'
        elif k.find('dgrad')>=0:
            return 'dgrad'
    return ph

################################################################################
## mxnet_kernel_to_marker()
################################################################################
def mxnet_kernel_to_marker(df):
    df['s_phase'] = df.apply(lambda row:kernel_to_phase(row),axis=1)
    return df

################################################################################
## prune()
##
## use prune_marker to filter out entries from a dataframe 
################################################################################
def prune(df):
    global prune_marker
    cret1 = df['s_layerName']==prune_marker
    cret2 = df['s_phase']=='bprop'
    df1 = df.index[cret1 & (cret2)]
    return df[df1[2]:] 

################################################################################
## dump_rows() 
##   
##   Walk all the rows in the table and prinstr
##############################################################################
def dump_rows(cursor=None, tbl_hdr=None, tbl_type=None):
    if cursor is None:
        print ("Error dump_rows: No cursor specified - exiting.")
        sys.exit(1)

    if tbl_hdr is None:
        print ("Error dump_rows: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows: No table type name specified- exiting.")
        sys.exit(1)

    ## Check the tbl_type - call the tbl specific dump function
    if (tbl_type == 'RUNTIME') or (tbl_type == 'DRIVER') :
        dump_rows_runtime_driver(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'NAME' :
        dump_rows_name(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'StringTable' :
        dump_rows_strings(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'MARKER' :
        dump_rows_marker(cursor, tbl_hdr, tbl_type)
    elif tbl_type == 'CONCURRENT_KERNEL' :
        dump_rows_conc_kernel(cursor, tbl_hdr, tbl_type)
    else:
        dump_rows_default(cursor, tbl_hdr, tbl_type)

    return
################################################################################
## dump_rows_default() 
################################################################################
def dump_rows_default (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type NAME "
    if cur is None:
        print ("Error dump_rows_default: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_default: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_default: No table type name specified - exiting.")
        sys.exit(1)

    for row in cur:
        if Debug:
            print ("DEFAULT {0} {1}".format(tbl_type, row))

    return
################################################################################
## dump_rows_name() 
################################################################################
def dump_rows_name (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type NAME "
    if cur is None:
        print ("Error dump_rows_name: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_name: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_name: No table type name specified - exiting.")
        sys.exit(1)

    # Get Row indexes
    if ('objectKind' in hdr)  and ('objectId' in hdr) and ('name')  :
        obj_kind_idx    = hdr['objectKind']
        obj_id_idx      = hdr['objectId']
        name_idx        = hdr['name']
    else :
        print ("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
        sys.exit(1)

    for row in cur:
        if Debug :
            print ("{0} {1} {2} {3}".format(tbl_type, row[name_idx], row[obj_kind_idx], row[obj_id_idx]))

    return
################################################################################
## dump_rows_strings() 
################################################################################
def dump_rows_strings(cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type StringTable "
    if cur is None:
        print ("Error dump_rows_strings: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_strings: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_strings: No table type name specified - exiting.")
        sys.exit(1)

    if ('_id_' in hdr)  and ('value' in hdr)  :
        str_id_idx   = hdr['_id_']
        str_name_idx = hdr['value']

    for row in cur:
        str_id   = row[str_id_idx]
        str_name = row[str_name_idx]
        if str_id not in string_hash:
            string_hash[str_id] = str_name
        if Debug:
            print ("{0} {1} {2}".format(tbl_type, row[str_id_idx], row[str_name_idx]))

    return

################################################################################
## dump_rows_conc_kernel() 
##
##  Note that the correlation ID in conc kernel maps to correlation ID in Runtime
##  Not always true in the reverse direction - Runtime covers more events
##  than just kernel
################################################################################
def dump_rows_conc_kernel(cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type CONCURRENT_KERNEL "
    global time_base
    # Get Row indexes
    if ('start' in hdr)  and ('end' in hdr) and ('registersPerThread' in hdr) and ('name' in hdr) and ('correlationId') and ('streamId')  :
        start_idx          = hdr['start']
        end_idx            = hdr['end']
        corr_id_idx        = hdr['correlationId']
        name_id_idx        = hdr['name']
        stream_id_idx      = hdr['streamId']
        regs_per_th_idx    = hdr['registersPerThread']
    else :
        print ("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))
        sys.exit(1)

    if Debug :
        print ("TblType ElapsedTime(ns) StartTime(ns) EndTime(ns) StreamId CorrId Regs Name")
    for row in cur:
        name_id         = row[name_id_idx]
        start_time      = row[start_idx]
        end_time        = row[end_idx]
        string_name     = string_hash[name_id]
        ## Get the first time stamp so we can subtract off the time since epoc
        if time_base == -1:
            time_base = start_time

        if Debug :
            time_base = 0
            print ("{0} {1} {2} {3} {4} {5} {6} {7}".format(tbl_type, end_time - start_time, start_time - time_base, end_time - time_base,  row[stream_id_idx], row[corr_id_idx], row[regs_per_th_idx], string_name))

    return

##
## decode_object_id()
##
def decode_object_id(obj_kind, obj_byte_array):
    '''
    Read in the object byte array
    The format is ProcID:ThreadID
    ProcID is 32 bits and threadID is 64 Bits
    The bytes in byte array are in reverse order
    '''
    pid   = 0
    th_id = 0
    if obj_kind != 2:
        print ("Error - unexpected obk_kind val -> {}, expecting 2".format(obj_kind))
        sys.exit(1)

    reverse_proc_id = obj_byte_array[:3]    ## Proc ID is 32 bits (4 bytes)
    reverse_th_id   = obj_byte_array[4:]    ## Thread ID is 64 bits - just take all the remaining bytes 

    pid   = int.from_bytes(reverse_proc_id, byteorder='little')
    th_id = int.from_bytes(reverse_th_id,   byteorder='little')

    #print ("ProcId -> {} Thread ID -> {}".format(pid, th_id))

    return [pid, th_id]
################################################################################
## dump_rows_marker() 
##
## Format for this table is 2 lines per event
##  First row - time stamp is the start time and the 'name' field is the string name of the event
##  Use the String Table to lookup the names - name to ID mapping - only valid for start of event row
##  The 'id' col is the event ID and it should be the same for both rows
##   2nd Row - Time stamp is stop time
##    Use 'id' to match up the start time stamp and event info
##  Additional info is available in the marker_data() table - use 'id' to lookup this data
##  'Category' is the field that is reported by the GUI
##  _id_,flags,timestamp,id,objectKind,objectId,name,domain
#    1,2,1509565664581882230,1,2,"^Z",3,0
#    2,4,1509565664620622854,1,2,"^Z",0,0

################################################################################
def dump_rows_marker (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for tbl type MARKER "

    global time_base
    global FwType
    marker_hash = {}
    if cur is None:
        print ("Error dump_rows_marker: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_marker: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_marker: No table type name specified - exiting.")
        sys.exit(1)

    # Get Row indexes
    if ('timestamp' in hdr)  and ('flags' in hdr) and ('id' in hdr) and ('name' in hdr) and ('objectKind' in hdr) and ('objectId' in hdr)  :
        ts_idx          = hdr['timestamp']
        flag_idx        = hdr['flags']
        event_id_idx    = hdr['id']
        name_id_idx     = hdr['name']
        object_kind_idx = hdr['objectKind']
        object_id_idx   = hdr['objectId']
    else :
        raise Exception ("Error - unexpected col names for tbl type {0} exiting...".format(tbl_type))


    if Debug:
        print ("TblType EventId NameId ElapsedTime(ns) StartTime(ns) EndTime(ns) LayerName LayerInstance ObjectKind ProcID ThreadID")
    for row in cur:
        if time_base == -1:
            time_base = row[ts_idx]
            break
        event_id  = row[event_id_idx]
        ## Save the name_id and the start time stamp for each event
        if event_id not in marker_hash :
            marker_hash[event_id] = [row[name_id_idx], row[ts_idx]]
            #print ("Adding event_id {0} to marker hash".format(event_id))
        else :
            name_id, start_time = marker_hash[event_id]
            elapsed_time = row[ts_idx] - start_time ## Elapsed time in ns
            string_net_name     = string_hash[name_id]
            net_name            = string_net_name
            long_name           = ""
            ## Try to figure out the framework that was used to generate the NVTX instrumentation
            if FwType is None:
                FwType = detect_fw_type(net_name)

            pat    = re.compile(r"(\S+)\s+(\S+)")
            a = re.match(pat, string_net_name)
            if a:
                net_name = a.group(1)
                long_name =a.group(2)
            #import pdb;pdb.set_trace()
            ## Convert ObjId into thread ID
            proc_id, thread_id = decode_object_id(row[object_kind_idx], row[object_id_idx])
            if Debug:
                print ("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}".format(tbl_type, event_id, name_id, elapsed_time, start_time - time_base, row[ts_idx] - time_base, net_name, long_name, row[object_kind_idx], proc_id, thread_id))
            if (row[flag_idx] !=4) :
                print ("Error - unexpected flag {0} for row {1}".format(row[flag_idx], row))
            del (marker_hash[event_id])

    return

################################################################################
##
## make_th_id_64bit() 
##   
################################################################################
def make_th_id_64bit(thread_id):
    '''
     For integer values that are < max_int32 and have non zero bit 31 they got converted to negative number by 
      this equation:  value - max_int32 = new_value (negative number)
     This code converts the negative number back to the positive int it is supposed to be : pos_int = neg_int + max_int32
    '''
    if thread_id < 0 :
        thread_id = max_int32 + thread_id
    return thread_id

################################################################################
## dump_rows_runtime_driver() 
##   
##   Walk all the rows in the table and print
##  runtime events map to different tables
##    Many events in runtime are cuda events - use the correlation ID to
##   lookup the CUDA event ID in the the table CUDA_EVENT
##   The events are numbered - I don't see a string equivalent to the number
##   The profiler must have an internal decoder for these events
##  - The other type of event is kernel event
##  - These events map to a different table
##    - So if the correlation ID is not found in cuda_event table
##    - Look in concurrent Kernel event table
##    - If the correlation ID matches - then check the Name ID field
##    - The name ID should return the string name of the event
##     - You can also compare time stamp info because the kernel table tracks it
################################################################################
def dump_rows_runtime_driver (cur=None, hdr=None, tbl_type=None):
    "Dump the contents of the sql cursor for TBL type RUNTIME or driver"
    global time_base
    if cur is None:
        print ("Error dump_rows_runtime_driver: No cursor specified - exiting.")
        sys.exit(1)

    if hdr is None:
        print ("Error dump_rows_runtime_driver: No col headers specified - exiting.")
        sys.exit(1)

    if tbl_type is None:
        print ("Error dump_rows_runtime_driver: No tbl type name specified - exiting.")
        sys.exit(1)

    # Get start time stamp
    if ('start' in hdr)  and ('end' in hdr) and ('threadId') and ('correlationId') and ('cbid')  :
        start_idx   = hdr['start']
        end_idx     = hdr['end']
        thread_idx  = hdr['threadId']
        corr_idx    = hdr['correlationId']
        cb_idx      = hdr['cbid']
    else :
        print ("Error: Col Hdrs {}", format(hdr))
        sys.exit(1)

    # Walk the cursor - print each row
    if Debug:
        print ("Start_time(ns) End_time(ns) Elapsed_time(ns) Thread_id Correlation_id Cb_id")
    num_rows = 0
    for row in cur:
        num_rows += 1
        ## Use driver to set up start time val
        if time_base == -1 and tbl_type == 'DRIVER':
            time_base = row[start_idx]
            #break
        thread = row[thread_idx]
        # For integer values that are < max_int32 and have non zero bit 31 they got converted to negative number by 
        thread = make_th_id_64bit(thread)
        if Debug:
            #print ("{0} {1} {2} {3} {4} {5} {6}".format(tbl_type, row[start_idx]-time_base, row[end_idx] - time_base, row[end_idx] - row[start_idx], thread, row[corr_idx], row[cb_idx]))
            print ("{0} {1} {2} {3} {4} {5} {6}".format(tbl_type, row[start_idx], row[end_idx], row[end_idx] - row[start_idx], thread, row[corr_idx], row[cb_idx]))

    ## Error checking - Driver and RUNTIME must be non empty
    if num_rows == 0:
        pass
        #raise Exception ("Table {} is empty".format(tbl_type))
    return

################################################################################
## get_tbl_names() 
##   
##   Walk all the rows in the table and print
################################################################################
def get_tbl_names (cur=None):
    "Dump the contents of the sql cursor"
    tbl_list = []
    if cur is None:
        print ("Error get_tbl_names: No cursor specified - exiting.\n")
        sys.exit(1)
    for row in cur:
        tbl_name = row[0]
        if Debug :
            print ("Tbl Name {0:s}".format(tbl_name))
        tbl_list.append(tbl_name)

    return tbl_list

################################################################################
## get_tbl_hdrs() 
################################################################################
def get_tbl_hdrs(cursor=None, display=True):
    tbl_hdr = {}   ## Hash table to map col header to index
    for idx, col in enumerate (cursor.description) :
        if(display) :
            print ("Col Header: {0} index {1}".format(col[0], idx))
        tbl_hdr[col[0]] = idx
    if(display) :
        ## Prtint the header in 1 row
        for idx, col in enumerate (cursor.description) :
            print ("{0} ".format(col[0]))
        print ("")
    return tbl_hdr
################################################################################
## process_driver_tbl() 
##   
##   Decode the DRIVER table
################################################################################
def process_tbl(tbl=None, cur=None, name=None):
    if tbl is None:
        print ("-Error- process_tbl: No tbl specified - exiting.\n")
        sys.exit(1)
    if cur is None:
        print ("-Error- process_tbl: No cursor specified - exiting.\n")
        sys.exit(1)
    if name is None:
        print ("-Error- process_tbl: No name specified - exiting.\n")
        sys.exit(1)

    pattern = re.compile(name)
    if pattern.search(tbl) :
        cmd_string = "select * from {};".format(tbl) 
        if Debug:
            print ("Executing sql cmd {}".format(cmd_string))
        cur.execute(cmd_string)   ## Need to use a tuple for variable sub- even though only passing 1 value 
        tbl_hdr = get_tbl_hdrs(cur, Debug)
        dump_rows(cur, tbl_hdr, name)

################################################################################
## get_marker_pandas_tbl_frame() 
##   
################################################################################
def get_marker_pandas_tbl_frame(tbl=None, cur=None) :
    """
    Returns pandas tbl frame for the marker table
    """
    query_string = "select name, id, timestamp, objectKind, objectId from {}".format(tbl)  
    tbl_hash     = {'name': [], 'name_id': [] , 'id': [], 'start_time': [], 'end_time': [], 'total_time': [],  'proc_id': [], 'thread_id': [] }  ## Hash used to create pandas frame
    cur.execute(query_string)
    tbl_list    = cur.fetchall()
    tbl_hdr     = get_tbl_hdrs(cur, False)
    name_id_idx = tbl_hdr['name']
    id_idx      = tbl_hdr['id']
    time_idx    = tbl_hdr['timestamp']
    kind_idx    = tbl_hdr['objectKind']
    obj_id_idx  = tbl_hdr['objectId']
    marker_hash = {}

    row_cnt     = 0

    for row in tbl_list:
        proc_id, thread_id  = decode_object_id(row[kind_idx], row[obj_id_idx])
        event_id            = row[id_idx]
        if event_id not in marker_hash :
            marker_hash[event_id] = [row[name_id_idx], row[time_idx]]
        else :
            name_id, start_time = marker_hash[event_id]
            end_time            = row[time_idx]
            elapsed_time        = end_time - start_time ## Elapsed time in ns
            marker_name         = string_hash[name_id]

            ## Populate table 
            tbl_hash['name_id'].append(name_id)
            tbl_hash['name'].append(marker_name)
            tbl_hash['id'].append(event_id)
            tbl_hash['start_time'].append(start_time)
            tbl_hash['end_time'].append(end_time)
            tbl_hash['total_time'].append(elapsed_time)
            tbl_hash['thread_id'].append(thread_id)
            tbl_hash['proc_id'].append(proc_id)
            row_cnt += 1
            del(marker_hash[event_id])
       
    ## Only create Pandas frame if row count is non zero
    if row_cnt > 0 :
        panda_frame = pd.DataFrame(tbl_hash)
    else :
        panda_frame  = None

    del tbl_hash

    return panda_frame


################################################################################
## get_runtime_pandas_tbl_frame() 
##   
################################################################################
def get_runtime_pandas_tbl_frame(tbl=None, cur=None) :
    """
    Copy a sql TBL into Pandas frame
    """
    tbl_hash     = {'start': [] , 'end': [], 'threadId': [], 'correlationId': [] }
    query_string = "select start, end, threadId, correlationId from {} ".format(tbl)
    cur.execute(query_string)
    tbl_list    = cur.fetchall()
    tbl_hdr     = get_tbl_hdrs(cur, False)
    start_idx   = tbl_hdr['start']
    end_idx     = tbl_hdr['end']
    th_idx      = tbl_hdr['threadId']
    cor_idx     = tbl_hdr['correlationId']
    for row in tbl_list :
        thread_id = make_th_id_64bit(row[th_idx])
        tbl_hash['start'].append(row[start_idx])
        tbl_hash['end'].append(row[end_idx])
        tbl_hash['threadId'].append(thread_id)
        tbl_hash['correlationId'].append(row[cor_idx])

    panda_frame = pd.DataFrame(tbl_hash)
    del tbl_hash
    return panda_frame
    
################################################################################
## time_stamp_to_duration() 
##   
################################################################################
def time_stamp_to_duration(ts_measured=None, ts_base=None, scale_factor=1) :
   '''
   Takes two time stamps and an optional scale factor and converts to a duration of time
   '''
   if scale_factor is 0:
       print("Error divide by 0 - exiting")
       sys.exit(1)

   if ts_measured is None or ts_base is None:
       print("Error bad arguments: either ts_measured or ts_base is Null")
       sys.exit(1)

   time  = (ts_measured - ts_base) / scale_factor
   return time

################################################################################
## name_lookup_by_id() 
##   
################################################################################
def link_kernel_to_dl_layer(cur=None, tbl_list=None, db_name=None, file_des=None) :
    """ 
    Walks the list of GPU kernel events and maps them to user level layer names
    defined in CPU CUDA runtime threads
    """
    global marker_tbl_index  
    global marker_tbl_size

    if cur is None  or tbl_list is None or db_name is None or file_des is None:
        print ("Error link_kernel_to_dl_layer: bad arguments - exiting.\n")
        sys.exit(1)


    kernel_events = []   ## Empty list - used to store the entire kernel tbl
    ns_to_ms_factor = 1000000
    tbl_str         = 'CONCURRENT_KERNEL'
    kernel_tbl      = get_tbl_name_from_type(tbl_str, tbl_list) 
    tbl_str         = 'RUNTIME'
    runtime_tbl     = get_tbl_name_from_type(tbl_str, tbl_list)
    tbl_str         = 'DRIVER'
    driver_tbl      = get_tbl_name_from_type(tbl_str, tbl_list)
    tbl_str         = 'MARKER'
    marker_tbl      = get_tbl_name_from_type(tbl_str, tbl_list)
    if kernel_tbl is None or runtime_tbl is None or marker_tbl is None or driver_tbl is None:
        print ("Error - Can't find table with substr {0:s} found".format(tbl_str))
        sys.exit(1)
    db_name         = os.path.basename(db_file)
    pivot_tbl_tag   = re.sub(r'[.]\w+', '', db_name)
    #query_string  = "select * from {}".format(kernel_tbl)
    query_string  = "select correlationId, start, end, name, gridX, gridY, gridZ, blockX, blockY, blockZ from {}".format(kernel_tbl)
    cur.execute(query_string)

    ## Store the whole table in memory
    kernel_events     = cur.fetchall()
    ## Get the runtime table and store in pandas frame
    runtime_tbl_frame = get_runtime_pandas_tbl_frame(runtime_tbl, cur)  
    ## Driver and Runtime tables are formatted the same 
    driver_tbl_frame  = get_runtime_pandas_tbl_frame(driver_tbl, cur)
    marker_tbl_frame  = get_marker_pandas_tbl_frame(marker_tbl, cur)
    print("Printer Marker table Frame")
    print(marker_tbl_frame)
    marker_tbl_size   = len(marker_tbl_frame.index)    ## Use index to get num_rows
    marker_tbl_index  = {}                             ## iterate from 0 to marker_tbl_size - move marker_tbl_index

    ## Store the table in a dict - Col headers are the keys - each val is a list, then pass the dict to Pandas to make 
    ## a frame Walk each row in the table - query the RUNTIME tbl for CPU start/end times
    report_tbl = {'s_layerName' : [], 's_layerOpName' : [], 's_layerType' : [], 's_phase' : [],  'd_cpuStartTimeMs' : [], 'd_cpuEndTimeMs' : [], 'd_cpuDurationMs' : [], 'GPUStartTime(ms)' : [], 'GPUEndTime(ms)' : [], 'd_gpuDurationMs' : [], 'l_CorrId' : [], 's_thread' : [], 's_kernel' : [], 's_ExperTag' : [], 's_GridXYZ' : [], 's_BlockXYZ' : []}
    file_des.write("s_layerName|s_layerType|s_phase|d_cpuStartTimeMs|d_cpuEndTimeMs|d_cpuDurationMs|GPUStartTime(ms)|GPUEndTime(ms)|d_gpuDurationMs|l_CorrId|s_thread|Kernel|s_ExperTag|s_GridXYZ|s_BlockXYZ")
    event_count = 0
    for kernel in kernel_events :
        event_count += 1
        #if event_count > 1000:
            #print ("Warning Exiting early for debug due to event count {} exceeded".format(event_count))
            #break;
        if HeartBeat and ((event_count % HeartBeat) == 0) :
            print("HeartBeat- {} kernel events processed out of {}".format(event_count, len(kernel_events)))

        ## Get the correlation ID
        ## Col names match the order used by query_string
        [corr_id, start_time, end_time, name_id, grid_x, grid_y, grid_z, block_x, block_y, block_z] = kernel
        ## Need to call map from name_id to name
        mangled_ker_name   = string_hash[name_id]
        ker_name           = demangle_kernel_name(mangled_ker_name)

        grid_coords  = "Grid-{}-{}-{}".format(grid_x, grid_y, grid_z)
        block_coords = "Block-{}-{}-{}".format(block_x, block_y, block_z) 

        # Use correlation ID to map kernel event to runtime cpu event
        cpu_start, cpu_end, thread_id         = get_tbl_event_by_corr_id(corr_id, runtime_tbl_frame, driver_tbl_frame)

        if marker_tbl_frame is None :
            marker_name, marker_start, marker_end = ["NA NA", cpu_start, cpu_end]
        else :
            ## Now find marker / range whose start time > cpu_start and end_time > cpu_end
            try:
                marker_name, marker_start, marker_end = get_tbl_marker_by_time_window(cpu_start, cpu_end, thread_id, marker_tbl_frame)
            except:
                marker_name, marker_start, marker_end = ["NA NA", cpu_start, cpu_end]


        ## Here the 2 fields from marker name mean different things depending on the framework
        ## For caffe2 the first field is the general layer name, the 2nd field is the layer instance
        ## For SFW (custome FW) - the first field is the phase naem (fprop/dgrad/wgrad) and 2nd
        ## is the layer instance name
        layer_instance = marker_name
        ###  @@@ Move this code into a function that figures out which fw thn returns type, phase,
        ## name
        layer_type, phase, layer_instance, layer_op  = decode_nvtx_marker_layer_string(layer_instance)

        ## Some thread IDs use the most significant bit in signed int32
        ## This converts the value to a positive integer
        if thread_id < 0 :
            thread_id = max_int32 + thread_id

        ## Convert time units ns -> ms
        marker_start_ms = time_stamp_to_duration(marker_start, time_base,   ns_to_ms_factor)
        marker_end_ms   = time_stamp_to_duration(marker_end,  time_base,    ns_to_ms_factor)
        marker_time_ms  = time_stamp_to_duration(marker_end,  marker_start, ns_to_ms_factor)
        gpu_start_ms    = time_stamp_to_duration(start_time,  time_base,    ns_to_ms_factor)
        gpu_end_ms      = time_stamp_to_duration(end_time,    time_base,    ns_to_ms_factor)
        gpu_time_ms     = time_stamp_to_duration(end_time,    start_time,   ns_to_ms_factor)

        th_id_str = "th_{}".format(thread_id)
        #print("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(layer_instance, layer_type, phase, marker_start_ms, marker_end_ms, marker_time_ms, gpu_start_ms , gpu_end_ms, gpu_time_ms, corr_id, th_id_str, ker_name, pivot_tbl_tag, grid_coords, block_coords), file=file_des)
        report_tbl['s_layerName'].append(layer_instance)
        report_tbl['s_layerOpName'].append(layer_op)
        report_tbl['s_layerType'].append(layer_type)
        report_tbl['s_phase'].append(phase)
        report_tbl['d_cpuStartTimeMs'].append(marker_start_ms)
        report_tbl['d_cpuEndTimeMs'].append(marker_end_ms)
        report_tbl['d_cpuDurationMs'].append(marker_time_ms)
        report_tbl['GPUStartTime(ms)'].append(gpu_start_ms)
        report_tbl['GPUEndTime(ms)'].append(gpu_end_ms)
        report_tbl['d_gpuDurationMs'].append(gpu_time_ms)
        report_tbl['l_CorrId'].append(corr_id)
        report_tbl['s_thread'].append(th_id_str)
        report_tbl['s_kernel'].append(ker_name)
        report_tbl['s_ExperTag'].append(pivot_tbl_tag)
        report_tbl['s_GridXYZ'].append(grid_coords)
        report_tbl['s_BlockXYZ'].append(block_coords)


    print("Finished processing NVVP file, creating Pandas Frame")
    ## Create Pandas data frame
    data_frame = pd.DataFrame(report_tbl)
    del report_tbl

    return data_frame

################################################################################
##   get_unique_tag_from_frame
##
##     Return a list of unique tags from a specific field in a pandas frame
##     Eg- for a col named 'Layers' - return a list of the unique layer names
################################################################################
def get_unique_tags_from_frame(field_name, pd_frame):
    unique_tags      = []
    tag_name_list    = pd_frame[field_name].tolist()
    
    for tag in tag_name_list:
        if tag not in unique_tags:
            unique_tags.append(tag) 
    return unique_tags

################################################################################
##  compute_ave_runtime()
##       count unique occurances of LayerName/Phase/Kernel - get the average of GPUDuration for each one
##        Return a new frame that just has GPUDuration Layer Info / Exper / Grid...
################################################################################
def compute_ave_runtime(pd_frame):
    ## foreach layer_name
    ##  filter - on layer_name & Phase
    ##    returns a table / frame - now need to filter on kernel name
    ##  Now you should have a frame that contains exactly the same number of rows as the
    ##  number of iterations 
    ##  Use data_frame['col_name'].mean() function to get the mean runtime
    ##  Create a new DF row using same col headings as old frame - take out the CPU start / end & GPU start end
    ##    Get row[0] from the frame used to compute mean - use all the other field values
    ##    Now you have a table w/ layer_name / layer_type / kernel / phase / Time (ms)
    ave_val_tbl  = {'s_layerName' : [], 's_layerOpName' : [], 's_layerType' : [], 's_phase' : [],  'd_cpuDurationMs' : [], 'd_gpuDurationMs' : [], 'l_CorrId' : [], 's_thread' : [], 's_kernel' : [], 's_ExperTag' : [], 's_GridXYZ' : [], 's_BlockXYZ' : []}
    unique_layer_names = get_unique_tags_from_frame('s_layerName', pd_frame)
    unique_phase_names = get_unique_tags_from_frame('s_phase', pd_frame)

    if Debug:
        print("Found {} unique layers ".format(len(unique_layer_names)))
    ## Now foreach unique layer_name - extract a frame that only has data for that layer name
    for phase in unique_phase_names:
        for layer_name in unique_layer_names:
            ## Print all rows in where col 'LayerName' == layer_name
            ## @@@ What if I throw out the first row?  That should get rid of cudnn_find
            layer_frame = pd_frame.loc[(pd_frame['s_layerName'] == layer_name) & (pd_frame['s_phase'] == phase)]
            ## Some layers are only available in some phases
            if(layer_frame.empty):
                continue
            #print("layer_frame {}".format(layer_frame))
            ## @@@ Almost done - if there is more than 1 kernel name - filter based on kernel
            ## names used in this layer
            unique_kernel_names = get_unique_tags_from_frame('s_kernel', layer_frame)
            for kernel in unique_kernel_names:
                kernel_frame      = layer_frame.loc[layer_frame['s_kernel'] == kernel]
                iterations        = len(kernel_frame.index)
                first_row_values  = kernel_frame.iloc[0]
                ave_gpu_runtime   = kernel_frame['d_gpuDurationMs'].mean()
                ave_cpu_runtime   = kernel_frame['d_cpuDurationMs'].mean()
                ## Now start building a new frame from a dictionary
                ave_val_tbl['d_gpuDurationMs'].append(ave_gpu_runtime)
                ave_val_tbl['d_cpuDurationMs'].append(ave_cpu_runtime)
                ave_val_tbl['s_kernel'].append(kernel)
                ave_val_tbl['s_layerName'].append(layer_name)
                
                if kernel.find("scalePackedTensor")>=0:
                    ave_val_tbl['s_phase'].append('wgrad')
                elif kernel.find("convertTensor")>=0:
                    ave_val_tbl['s_phase'].append('wgrad')
                elif kernel.find("computeOffsetsKernel")>=0:
                    ave_val_tbl['s_phase'].append('dgrad')
                elif kernel.find("computeBOffsetsKernel")>=0:
                    ave_val_tbl['s_phase'].append('dgrad')
                else:    
                    ave_val_tbl['s_phase'].append(phase)
                ave_val_tbl['s_BlockXYZ'].append(first_row_values['s_BlockXYZ'])
                ave_val_tbl['l_CorrId'].append(first_row_values['l_CorrId'])
                ave_val_tbl['s_ExperTag'].append(first_row_values['s_ExperTag'])
                ave_val_tbl['s_GridXYZ'].append(first_row_values['s_GridXYZ'])
                ave_val_tbl['s_layerType'].append(first_row_values['s_layerType'])
                ave_val_tbl['s_layerOpName'].append(first_row_values['s_layerOpName'])
                ave_val_tbl['s_thread'].append(first_row_values['s_thread'])
                if Debug:
                    print("Layer {} Phase {} Kernel {} GPU-Time {} CPU-Time {} iterations {}".format(layer_name, phase, kernel, ave_gpu_runtime, ave_cpu_runtime, iterations))

    ## Now put everything into a frame 
    ave_val_frame = pd.DataFrame.from_dict(ave_val_tbl)
    return ave_val_frame

################################################################################
## strip_layer_name() 
##   
################################################################################
def strip_layer_name (l_name) :
    '''
    Strip the preamble off of a layer name 
    Return stripped down name and the part that
    was removed
    '''
    l_inst_tag = l_name
    l_inst     = l_name
    ## Look for eg.  conv2_1_2_bn, conv1_3_shortcut_bn, conv1_bn
    pattern = re.compile(r"(conv\d+_\d+_\d+|conv\d+_\d+|conv\d+)_(\S+)")
    res     = re.search(pattern, l_name)
    if res is not None:
        l_inst_tag     = res.group(1)
        l_inst         = res.group(2)
        if Debug:
            print("Layer -> {} Layer inst tag -> {} - layer inst -> {}".format(l_name, l_inst_tag, l_inst, l_name))

    return l_inst, l_inst_tag

################################################################################
## get_pyt_exporterlayer_type_from_name() 
##   
################################################################################
def get_pyt_exporter_layer_type_from_name(marker_str):
    pat = re.compile(r"(NA)\s+(NA)")
    res = re.search(pat, marker_str)
    if res is not None:
        l_name  = res.group(1)
        l_type  = res.group(2)
        l_phase = 'Fprop'
        return l_type, l_phase, l_name
    #print("Marker str {}".format(marker_str))
    l_name, l_type, l_phase = re.split(',', marker_str)
    return l_type, l_phase, l_name

################################################################################
## get_pytorch_layer_type_from_name() 
##   
################################################################################
def get_pytorch_layer_type_from_name(layer_instance):
    l_name  = layer_instance
    l_phase = "Fprop"

    ## Remove the pytorch tag eg N5torch8autograd3
    l_name  = re.sub(r"^N\d+\w+\d+\w+\d+", "", l_name)
    l_type  = l_name

    pat = re.compile(r"(\w+)Forward")
    res = re.search(pat, l_name)
    if res is not None:
        l_type  = res.group(1)
        l_phase =  "Fprop"
        return [l_type, l_phase, l_name]
    
    pat = re.compile(r"(\w+)GradE")
    res = re.search(pat, l_name)
    if res is not None:
        l_type  = res.group(1)
        l_phase = "Dgrad"
        return [l_type, l_phase, l_name]

    pat = re.compile(r"(\w+)[Bb]ackward[E]*$")
    res = re.search(pat, l_name)
    if res is not None:
        l_type  = res.group(1)
        l_phase = "Wgrad"
        return [l_type, l_phase, l_name]

    return [l_type, l_phase, l_name]
################################################################################
## get_tensorflow_layer_info_with_map() 
##   
################################################################################
def get_tensorflow_layer_info_with_map(layer_instance):
    '''
    Tensorflow graph groups operators into clusters
    The graph_info_map maps the unique operator name to a cluster name
    If there is no cluster associated with the operator name - then just default to 
    get_tensorflow_layer_type_from_name()
    '''
    global graph_info_map
    l_type       = layer_instance
    l_phase      = "Fprop"
    l_name       = l_type
    op_name      = None
    unique_name  = layer_instance

    res = re.search(r"(\S+):\s+(\S+)", layer_instance)
    if res :
        op_name     = res.group(1)
        unique_name = res.group(2)
    else :
        raise Exception("Unexpected NVTX marker format {} for Tensorflow framework".format(layer_instance))

    l_type    = op_name
    l_name    = unique_name
    l_op_name = unique_name
    ## @@@ Only doing Fprop for now
    ## Use graph info to find attributes about each operator
    if unique_name in graph_info_map:
        cluster = graph_info_map[unique_name]
        l_name  = cluster
        #print("Layer instance {} is in cluster {}".format(unique_name, cluster))

    if re.search(r"gradients[/]", l_name):
        ## At this point it could be either wgrad or dgrad - default to wgrad
        ## wgrad includes ConvBackpropFilter and ops like FusedBatchNormGrad
        ## Dgrad is only for ops that use the chain rule and split weights/data gradient
        l_phase = "Dgrad"
        if re.search(r"BackpropFilter", l_type):
            l_phase = "Wgrad"
        l_type = re.sub("Backprop.*$", "", l_type)
        l_type = re.sub("Grad.*$", "", l_type)

    elif re.search(r"train.update_model", l_name):
        l_phase = "Wupdate"

    elif re.search(r"(cross_entropy|Loss|Momentum)", unique_name): 
        l_phase = "Wgrad"

    return [l_type, l_phase, l_name, l_op_name]
################################################################################
## get_tensorflow_layer_type_from_name() 
##   
################################################################################
def get_tensorflow_layer_type_from_name(layer_instance):
    '''
    tensorflow layer instance is a '/' separated string like this
    transformer/parallel_0_5/transformer/symbol_modality_33952_512_2/shared/Reshape_1/shape/2
    '''
    l_type       = layer_instance
    l_phase      = "Fprop"
    l_name       = l_type
    grad_type    = None

    ## Look for back prop layers key word 'training'
    pat = re.compile(r"training[/](\S+)")
    res = re.match(pat, layer_instance)
    if res is not None :
        l_name  = res.group(1)
        l_type  = l_name
        ## Set phase to Wgrad - dgrad is indicated by the string 'gradient'
        l_phase = "Wgrad"
        ## First - detect whether or not backprop
        ## Take the layer name string - and widdle it down
        ## Optional string between gradients and _grad
        pat = re.compile(r"gradients[/]*(\S*[/]+(\w+)_grad[/](\w+))")
        res = re.search(pat, layer_instance)
        if res is not None:
            l_name    = res.group(1)
            grad_type = res.group(2)
            l_type    = res.group(3)
            l_phase = "Dgrad"
            l_type  = "{}_{}".format(grad_type, l_type)

    ## @@@ Take the field right before layer eg encoder/decoder and make it part of the Phase
    ## remove it from layer name
    ## eg Phase = 'Fprop,encoder' or 'Dgrad,decoder'
    ## This should cover all the layers - so every layer now starts w/ layer_
    pat = re.compile(r"(\w+)[/](layer_\w+\S+)")
    res_ = re.search(pat, l_name)
    if res_ is not None:
        l_phase = "{},{}".format(l_phase, res_.group(1))
        l_name  = res_.group(2)

    ## Get the sub-layer info from the name
    pat = re.compile(r"layer_\w+[/](\w+[/]\w+)")
    res_ = re.search(pat, l_name)
    if res_ is not None:
        l_type = res_.group(1)
    else:
        ## This means the layer name does not have the pattern layer_[0-9] in it
        pat = re.compile(r"\w+[/]\w+[/]\w+[/](\S+)")
        res = re.match(pat, l_name)
        if res is not None:
            l_name = res.group(1)
        l_name = re.sub(r"^body[/]", "", l_name)
        l_type = l_name

    ## Convert strings like this sub_1_Sum_1 to sub_Sum
    l_type = re.sub(r"_\d+", "", l_type) 
    l_type = re.sub(r"\d+",  "", l_type) 
    return [l_type, l_phase, l_name]
################################################################################
## get_caffe2_layer_type_from_name() 
##   
################################################################################
def get_caffe2_layer_type_from_name(layer_operator=None, layer_instance=None):
    '''
    caffe2 has 2 fields to describe the layer - these fields have 3 things
    encoded in the names
    1. Layer instance name
    2. Layer type
    3. Phase - Fprop / Dgrad / Wgrad
    4. Other - weight initialization algo etc
    Under layer type - include param initialization - ParamInit
    Eg BatchNorm -> riv -> running_inv_var, bn_rm ->running_mean, bn_b -> bias bn_s -> scale
    Just make a layer type named Weight Init
    Or - use ConstantFill / MSRAFill to indicate weight init
    '''
    l_type       = layer_operator
    l_phase      = layer_operator
    l_name       = layer_instance
    
    if layer_operator is None or layer_instance is None :
       print("Error get_caffe2_layer_type_from_name - Bad args - exiting...")
       sys.exit(1)

    pattern = re.compile(r"(\w+)_w_grad")
    res  = re.search(pattern, layer_instance)
    extra = "NA"
    if res is not None:
        l_name = res.group(1)
        l_name, l_tag = strip_layer_name(l_name)
        l_phase = "Wgrad"
        ## First looks for FilterGradient eg ConvFilterGradient, then looks for just Gradient eg FCGradient
        res = re.search(r"(\w+)(Gradient)", layer_operator)
        if res is not None:
            l_type = res.group(1)
            extra  = res.group(2)
            res = re.search(r"(\w+)Filter", l_type)
            if res is not None:
                l_type = res.group(1)
        if Debug:
            print ("Found {} phase for layer_instance {} layer_name {} layer_type {} extra {} ".format(l_phase, layer_instance, l_name, l_type, extra))
        return [l_type, l_phase, l_name]

    ## @@@ What about conv_bn ConvDataGradient conv5_3_1_bn_grad - Should layer type be batch norm?
    pattern = re.compile(r"(\w+)_grad")
    res  = re.search(pattern, layer_instance)
    if res is not None:
        l_name        = res.group(1)
        l_name, l_tag = strip_layer_name(l_name)
        l_phase       = "Dgrad"
        ## First looks for DataGradient eg ConvDataGradient, then looks for just Gradient eg FCGradient
        res = re.search(r"(\w+)(Gradient)", layer_operator)
        if res is not None:
            l_type = res.group(1)
            extra  = res.group(2)
            res = re.search(r"(\w+)Data", l_type)
            if res is not None:
                l_type = res.group(1)
        if Debug:
            print ("Found {} phase for layer_instance {} layer_name {} layer_type {} extra {}".format(l_phase, layer_instance, l_name, l_type, extra))

        return [l_type, l_phase, l_name]

    ## Fprop - any layer that isn't a gradient layer
    l_name  = layer_instance
    l_phase = "Fprop"
    l_type  = layer_operator 
    if Debug:
        print ("Found {} phase for layer_instance {} layer_name {} layer_type {} ".format(l_phase, layer_instance, l_name, l_type))
    
    return [l_type, l_phase, l_name]
################################################################################
## detect_fw_type() 
##   
################################################################################
def detect_fw_type(net_name):
    fw_type  = None
    if re.match(r"\w+[/]\S+", net_name) :
        fw_type = "FW_TENSOR_FLOW"
    elif re.match(r"N\d+torch", net_name) :
        fw_type = "FW_PYTORCH"
    elif re.match(r".+[(].+", net_name) :
        fw_type = "FW_TENSORRT"
    return fw_type

################################################################################
## process_mxnet() 
################################################################################
mxnet_phase = 'fprop'
mxnet_print_interval = 1000

def rename_mxnet_layer(old_name):
    '''
            1) Converts stage1_unit1_conv3 to res2a_branch2c
            2) Handle Case (1): add_stage1_unit1_relu3_fprop    add_relu    fprop   to res2a_add_relu_branch2c
            3) Handle Case (2): conv0 --> conv1
            4) handle case (3): fc1 --> fc 
            5) Handle case (4): res0_bn --> bn_conv1 
    '''
    cmap = {1:'a',2:'b',3:'c',4:'d',5:'e',6:'f'}
    dmap = {1:'2a',2:'2b',3:'2c'}
    ln = old_name

    reg = re.compile(r"stage(?P<st>\d+)_unit(?P<unit>\d+)_(?P<type>[a-zA-Z]+)(?P<typenum>\d+)")
    reg2 = re.compile(r"stage(?P<st>\d+)_unit(?P<unit>\d+)_(?P<type>[a-zA-Z0-9]+)")
    reg3 = re.compile(r"add_stage(?P<st>\d+)_unit(?P<unit>\d+)_(?P<type>[a-zA-Z]+)(?P<typenum>\d+)")

    m = re.match(reg,ln)
    m2 =re.match(reg2,ln)
    m3 =re.match(reg3,ln) 
    name = ln

    if m :
        if m.group('type')=='conv':
            name = "res{}{}_branch{}".format(int(m.group('st'))+1,cmap[int(m.group('unit'))],dmap[int(m.group('typenum'))]) 
        elif m.group('type')=='bn':
            name = "res{}{}_bn_branch{}".format(int(m.group('st'))+1,cmap[int(m.group('unit'))],dmap[int(m.group('typenum'))])
    if m2:
        if m2.group('type')=='conv1sc':
            name = "res{}{}_branch1".format(int(m2.group('st'))+1,cmap[int(m2.group('unit'))])
        elif m2.group('type')=='sc':
            name = "res{}{}_bn_branch1".format(int(m2.group('st'))+1,cmap[int(m2.group('unit'))]) 
    #case1
    if m3:
        name = "res{}{}_add_relu_branch{}".format(int(m3.group('st'))+1,cmap[int(m3.group('unit'))],dmap[int(m3.group('typenum'))]) 
    #case2
    if name.find('conv0')>=0:
        name = name.replace('conv0','conv1') 
    #case3
    if name.find('fc1000')>=0:
        name = name.replace('fc1000','fc')
    #case 4
    if name.find('res0_bn')>=0:
        name = name.replace("res0_bn","bn_conv1")                                                                                                                                                   
    
    return name

def process_mxnet(layer_instance):
    global mxnet_phase
    global mxnet_print_interval
    layer_instance = layer_instance.strip('[')  
    layer_instance = layer_instance.rstrip(']') 
    if mxnet_print_interval==0:
        pass
    layer_regex = re.compile(r"(?P<type>[a-zA-Z_]+){name=(?P<name>[a-zA-Z0-9_]+)")
    m = re.match(layer_regex,layer_instance)
    if m :
        lname = m.group('name')
        lname = lname.replace("_backward","")
        ltype = m.group('type')
        if ltype.find("_backward_")>=0:
            ltype = ltype.replace("_backward_","")
            lphase = 'bprop'
            if layer_instance.find('dgrad')>=0:
                lphase = 'dgrad'
            elif layer_instance.find('wgrad')>=0:
                lphase = 'wgrad'
            if ltype=="BatchNorm":
                lpahse = "dgrad"
        else:
            lphase = 'fprop'
        if ltype.find("BatchNorm")>=0 and lphase=='bprop':
            lphase = "dgrad"
        elif ltype.find("FullyConnected")>=0 and lphase=='bprop':
            lphase = "dgrad"
        elif ltype.find("Pooling")>=0 and lphase=='bprop':
            lphase = "dgrad"
        if mxnet_print_interval==0:
            mxnet_print_interval = 1000
            #print(ltype,lphase,lname,ltype)
            #print(ltype,lphase,rename_mxnet_layer(lname),ltype)
        mxnet_phase = lphase
        mxnet_print_interval = mxnet_print_interval - 1
        #return(ltype,lphase,lname,ltype)
        return(ltype,lphase,rename_mxnet_layer(lname),ltype)
    else:
        #print(layer_instance)
        return ("UNKNOWN",mxnet_phase,layer_instance,"UNKNOWN")

################################################################################
## get_layer_type_from_name() 
##   
################################################################################
def decode_nvtx_marker_layer_string(layer_instance):
    '''
    Each framework has a different way of encoding the layer info in the NVTX marker string
    So far Caffe2, KNF, TensorFlow, are supported
    Also - there is format for decoding the layer for cases where we control the NVTX string in the
    workload
    '''
    global FwType
    
    if FwType=="FW_MXNET":
        return process_mxnet(layer_instance)

    l_op_name = None
    if FwType == "FW_TENSORRT":
        l_type, l_phase, l_name  = get_tensorrt_layer_type_from_name(layer_instance)
    elif FwType == "FW_PYTORCH":
        l_type, l_phase, l_name = get_pyt_exporter_layer_type_from_name(layer_instance)
        #l_type, l_phase, l_name = get_pytorch_layer_type_from_name(layer_instance)
    elif FwType == "FW_TENSOR_FLOW":
        if graph_input_file:
            l_type, l_phase, l_name, l_op_name =  get_tensorflow_layer_info_with_map(layer_instance)
        else:
            l_type, l_phase, l_name =  get_tensorflow_layer_type_from_name(layer_instance)
    elif FwType == 'FW_CAFFE2':
        res = re.search(r"(\w+)\s+\((\S+)\)", layer_instance)
        operator = res.group(1)
        layer_instance = res.group(2)
        ## Reset layer_type, phase, layer_name for caffe2
        ## layer_name return value may be redundant
        l_type, l_phase, l_name = get_caffe2_layer_type_from_name(operator, layer_instance)
    else:
        l_type, l_phase, l_name  = get_layer_type_from_name(layer_instance)

    return [l_type, l_phase, l_name, l_op_name]
################################################################################
## get_tensorRT_layer_type_from_name() 
##   
################################################################################
def get_tensorrt_layer_type_from_name(name=None) :
    """
    parse the long form layer name for tensorRT
    """
    layer_type = 'UNK'
    layer_name = "UNK"
    Phase = 'Fprop'
    if name is None :
        print("Error get_layer_type_from_name - Bad args - exiting...")
        sys.exit(1)
    layer = re.match(r"([^(]+)[(](.+)",name)
    if layer:
        layer_name, layer_type = layer.group(1),layer.group(2)
    else:
        layer_name = name
    return layer_type, Phase, layer_name
################################################################################
## get_layer_type_from_name() 
##   
################################################################################
def get_layer_type_from_name(name=None) :
   """
   get the layer type from the long form layer name
   """
   if name is None :
       print("Error get_layer_type_from_name - Bad args - exiting...")
       sys.exit(1)
   #print(name)
   phase, layer_name = name.split(' ')
   layer_type        = layer_name    

   ## @@@ For new NVTX - make the convension 'Phase LayerType,UniqueLayerName'
   pattern = re.compile(r"([a-zA-Z0-9]+),(\S+)")
   res = re.match(pattern, layer_name)
   if res is not None:
       layer_type = "{}".format(res.group(1))
       layer_name = "{}".format(res.group(2))
       return layer_type, phase, layer_name

   '''
   ## @@@ For Deep Bench - Remove this - make Deep Bench follow 'Phase Type,UniqueName' pattern
   pattern = re.compile(r"(Conv_\d+x\d+)")
   res = re.match(pattern, layer_name)
   if res is not None:
       layer_type = "{}".format(res.group(1))
       return layer_type, phase, layer_name
   '''

   ### All remaining pattern matches are there to support KNF naming convention

   pattern = re.compile(r"layer_\d+_\d+_(\w+)")
   res = re.match(pattern, layer_name)
   if res is not None:
       layer_type = "{}".format(res.group(1))
       return layer_type, phase, layer_name

   ## Look for res_branch_relu tag
   #pattern = re.compile(r"res\w+_branch\w+_(relu)")
   pattern = re.compile(r"res\w+[_]+(relu)")
   res = re.match(pattern, layer_name)
   if res is not None:
       layer_type = "{}".format(res.group(1))
       return layer_type, phase, layer_name

   ## Look for res_branch tag
   pattern = re.compile(r"res\w+_branch\w+")
   res = re.match(pattern, layer_type)
   if res is not None:
       layer_type = "conv"
       return layer_type, phase, layer_name

   ## Look for bn_branch tag
   pattern = re.compile(r"(bn)\w+_branch\w+")
   res = re.match(pattern, layer_type)
   if res is not None:
       layer_type = "{}".format(res.group(1))
       return layer_type, phase, layer_name

   pattern = re.compile(r"res\d+[a-f]")
   res     = re.match(pattern, layer_type)
   if res is not None:
       if Debug:
           print ("Found elt layer type from {}".format(layer_type)) 
       layer_type = "elt" 
       return layer_type, phase, layer_name

   # Get rid of numbers
   layer_type = re.sub(r"\d+", "", layer_type)

   # Special case - conv_expand - is a conv layer
   pattern = re.compile(r"(\w+)_expand")
   res = re.match(pattern, layer_type)
   if res is not None:
       layer_type = "{}".format(res.group(1))
       return layer_type, phase, layer_name

   ## Look for bn_conv - V1 prototxt format has bn as first field V2 has it as 2nd field
   pattern = re.compile(r"bn_(conv)")
   res = re.match(pattern, layer_type)
   if res is not None:
       layer_type = "bn"
       return layer_type, phase, layer_name

   ## Look for compound layer names - use the 2nd field for the name
   layer_type = re.sub(r".*_(\w+)", "\g<1>", layer_type)


   return layer_type, phase, layer_name
#################################################################################
## demangle_kernel_name() 
##   
################################################################################
def demangle_kernel_name(mangled_name=None) :
    """
    Kernel names are mangled use c++filt to get human readable names
    """
    global kernel_hash

    if mangled_name not in kernel_hash :
        ret      = subprocess.run(['c++filt', mangled_name], stdout=subprocess.PIPE)
        new_name = ret.stdout.decode("utf-8").strip()
        kernel_hash[mangled_name] = new_name
        if Debug :
            print ("C++filt kernel name -> {}".format(new_name))
    else :
        new_name = kernel_hash[mangled_name]

    return new_name
################################################################################
## get_tbl_name_from_type() 
##   
################################################################################
def get_tbl_name_from_type(tbl_type=None, tbl_list=None) :
    """
    Return full table name that matches the tbl_type substring
    """
    tbl_name = None
    if tbl_type is None:
        print ("Error get_tbl_name_from_pattern: No tbl_type specified - exiting.\n")
        sys.exit(1)

    if tbl_list is None:
        print ("Error get_tbl_name_from_pattern: No tbl_list specified - exiting.\n")
        sys.exit(1)


    ## Walk the list of tbls - return the one that has substring tbl_type
    for tbl in tbl_list :
        pattern = re.compile(tbl_type)
        if pattern.search(tbl) :
            tbl_name = tbl
            break

    return tbl_name

################################################################################
## get_tbl_marker_by_time_window() 
##
##  @@@ This function is slow - sql look ups seem to take really long
##  Try using Pandas instead - create a frame
##  @@@ Instead of doing a query every time this is called - should read
##  in the table once - then keep a pointer to the last row that was selected
##  start the next search from the row pointer - this should save a lot of time
##  because each search will only be a couple iterations
##  Search by time stamp
##  return the fields
##   
################################################################################
def get_tbl_marker_by_time_window(cpu_start=None, cpu_end=None, thread_id=None, pd_frame=None, tbl_size=None) :
    """
        Find the marker / range whose start and end times cover the cpu event 
        start and end times passed in
    """
    global marker_tbl_index  
    global marker_tbl_size
    global time_base

    if cpu_start is None or cpu_end is None or pd_frame is None or thread_id is None:
        print ("get_tbl_marker_by_time_window: Bad args - exiting ")
        sys.exit(1)

    marker_end   = None
    pd_marker_id = None

    '''
    ## Initialize index first time it sees thread_id
    if thread_id not in marker_tbl_index:
        marker_tbl_index[thread_id] = 0

    row_index = marker_tbl_index[thread_id]
    for row in range(row_index, marker_tbl_size):
        marker_tbl_index[thread_id] += 1
        frame_row = pd_frame.iloc[row]
        if frame_row['timestamp'] > cpu_end and frame_row['thread_id'] == thread_id:
            pd_marker_id = frame_row['id']
            marker_end   = frame_row['timestamp']
            break

    if (marker_end is None or pd_marker_id is None) and (row_index < marker_tbl_size):
        raise Exception("Query failed for timestamp  > {} and thread_id == {} row index {} timestamp (ms) {}".format(cpu_end, thread_id, row_index, (cpu_end-time_base)/1000000)) 
    '''

    ## Get the first entry whose time stamp is > end
    ## record the ID - then do a 2nd query that returns name when ID == id from prev query
    #query_string     = "(timestamp > {}) & (thread_id == {})".format(cpu_end, thread_id)
    #tmp_frame        = pd_frame.query(query_string)
    tmp_frame        = pd_frame[(pd_frame['end_time'] > cpu_end) & (pd_frame['thread_id'] == thread_id)]
    if tmp_frame.empty:
        raise Exception("Query failed for timestamp  > {} and thread_id == {}".format(cpu_end, thread_id)) 
    pd_marker_id     = tmp_frame['id'].iat[0]
    marker_end       = tmp_frame['end_time'].iat[0]
    marker_start     = tmp_frame['start_time'].iat[0]
    pd_name_id       = tmp_frame['name_id'].iat[0]
    marker_name      = string_hash[pd_name_id]
    
    ## 2nd Query using ID to get name and marker start time
    #query_string     = "id == {}".format(pd_marker_id)
    #tmp_frame        = pd_frame.query(query_string)
    #pd_name_id       = tmp_frame['name_id'].iat[0]
    #marker_name         = string_hash[pd_name_id]
    #marker_name      = tmp_frame['name'].iat[0]
    #marker_start     = tmp_frame['timestamp'].iat[0]
    ## Reset the marker tbl index to the start time that maps to this id
    #marker_tbl_index[thread_id] = tmp_frame.index[0]

    if(Debug) :
        print ("Marker name {} start {} end {} ".format(marker_name, marker_start, marker_end))
    return marker_name, marker_start, marker_end

################################################################################
## get_tbl_event_by_corr_id() 
##   
################################################################################
def get_tbl_event_by_corr_id(corr_id=None, pd_frame=None, driver_frame=None) :
    if corr_id is None or pd_frame is None:
        print ("Error get_runtime_event_by_corr_id: missing argument - exiting.\n")
        sys.exit(1)
    ## use panda frame instead of sql query
    query_string = "correlationId == {}".format(corr_id)
    tmp_frame = pd_frame.query(query_string)
    if tmp_frame.empty :
        if driver_frame is not None:
            tmp_frame = driver_frame.query(query_string)
            if tmp_frame.empty:
                raise Exception ("Query {} failed for lookup in RUNTIME and DRIVER table ".format(query_string))

    start     = tmp_frame['start'].iat[0]
    end       = tmp_frame['end'].iat[0]
    thread_id = tmp_frame['threadId'].iat[0]

    return [start, end, thread_id]
################################################################################
## name_lookup_by_id() 
##   
################################################################################
def tbl_name_lookup_by_id(cur=None, name_id=None) :
    if name_id is None:
        print ("Error name_lookup_by_id - no name specified - exiting...")
        sys.exit(1)

    if cur is None:
        print ("Error process_runtime_tbl: No cursor specified - exiting.\n")
        sys.exit(1)
    
    query_string = "select value from StringTable where _id_={0}".format(name_id)

    return

################################################################################
## process_runtime_tbl() 
##   
##   Decode the RUNTIME table
################################################################################
def process_runtime_tbl(tbl=None, cur=None):
    if tbl is None:
        print ("Error process_runtime_tbl: No tbl specified - exiting.\n")
        sys.exit(1)
    if cur is None:
        print ("Error process_runtime_tbl: No cursor specified - exiting.\n")
        sys.exit(1)

    pattern = re.compile('RUNTIME')
    if pattern.search(tbl) :
        cmd_string = "select * from {};".format(tbl) 
        print ("Executing sql cmd {}".format(cmd_string))
        cur.execute(cmd_string)   ## Need to use a tuple for variable sub- even though only passing 1 value 
        tbl_hdr = get_tbl_hdrs(cur, Debug)
        dump_rows(cur, tbl_hdr, 'RUNTIME')


    return
################################################################################
## Main program
################################################################################

## Call the functions
parse_cmd_line()
output_fd = open_ouput_file()
if graph_input_file:
    with  open(graph_input_file, "r") as yml_fd :
        graph_info_map = load(yml_fd, Loader=Loader) 

## Make 2 new sheets - 1 has the raw data 1 has average values
frame_list         = []
ave_val_frame_list = []
for db_file in db_file_list :
    pd_frame = read_db_file(db_file, output_fd);
    frame_list.append(pd_frame)
    if ComputeAverage :
        pd_ave_val_frame = compute_ave_runtime(pd_frame)
        ave_val_frame_list.append(pd_ave_val_frame)

    ## Make 1 worksheet per experiment
    panda_sheet  = os.path.basename(db_file)
    # Drop the file extension
    panda_sheet  = re.sub(r"[.]\w+", "", panda_sheet)
    if(len(panda_sheet) > MAX_EXCEL_SHEET_LEN) :
        #raise Exception("worksheet name {} too long - Max num chars {} !".format(panda_sheet,  MAX_EXCEL_SHEET_LEN))
        panda_sheet = panda_sheet[0:MAX_EXCEL_SHEET_LEN-1]
    pd_frame.to_excel(excel_writer, panda_sheet)

## Make a combined pivot table + a 2nd sheet that takes average over all iterations
panda_sheet          = 'combined_tbl'
panda_ave_val_sheet  = panda_sheet + "_ave"
pivot_tbl_frame = pd.concat(frame_list)
## Combine all the frames into 1 
pivot_tbl_frame.to_excel(excel_writer, panda_sheet)

if ComputeAverage:
    ave_val_pivot_tbl_frame = pd.concat(ave_val_frame_list)
    ave_val_pivot_tbl_frame.to_excel(excel_writer, panda_ave_val_sheet)

## Close the xcel sheet
excel_writer.save()
if pivot_tbl is not None:
    output_fd.close()

## Output json
for db_file in db_file_list :
    
    now = datetime.now()
    # timestamp returned by python is in secs -- Kibana needs ts in ms
    ts_created = str(int(datetime.timestamp(now)*1000))
    s_ci_job_id = os.environ['CI_JOB_ID']
    l_batch_size = os.environ['BATCHSIZE']
    try:
        s_network = os.environ['NETWORK']
    except:
        s_network = "UNK"
    datadict = ave_val_pivot_tbl_frame.to_dict(orient='records')
    for i in datadict:
        f  = open("{}_{}_{}.json".format(db_file,i['s_layerName'],i['l_CorrId']),'w')
        i['ts_created']  = ts_created 
        i['s_ci_job_id'] = s_ci_job_id 
        i['l_batch_size'] = l_batch_size 
        i['s_network'] = s_network 
        kibana_json_string = json.dumps(i)
        f.write(kibana_json_string)
        f.close()
