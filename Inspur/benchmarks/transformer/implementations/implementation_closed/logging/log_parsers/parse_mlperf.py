# Copyright 2018 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import csv
import sys
import re
import argparse
from datetime import datetime, timedelta

# Third-party modules
import pytz
import plotly.graph_objects as pgo

# Global Variables
#   g_*_tz 			  : to help with host/dut TZ differences
#   g_power*td		  : manual tweaking of timedelta, adding or subtracting by seconds from the power log timestamps
#   g_power_window*   : how much time before (BEGIN) and after (END) loadgen timestamps to show data in graph
#   g_power_stats*    : when to start (after BEGIN) and end (before END) loadgen timestamps to calculate statistical data of graph
g_power_tz     			 = None # pytz.timezone( 'US/Pacific' )
g_loadgen_tz   			 = None # pytz.utc
g_power_add_td 			 = timedelta(seconds=3600)
g_power_sub_td  		 = timedelta(seconds=0)
g_power_window_before_td = timedelta(seconds=30)
g_power_window_after_td  = timedelta(seconds=30)
g_power_stats_begin_td   = timedelta(seconds=3)
g_power_stats_end_td     = timedelta(seconds=3)

# Check command-line parameters and call respective functions
def main():

	m_args = f_parseParameters()

	if( m_args.loadgen_in != "" ):
		f_parseLoadgen( m_args.loadgen_in, m_args.loadgen_out )

	if( m_args.power_in != "" ):
		f_parsePowerlog( m_args.power_in, m_args.power_out )

	if( m_args.graph ):
		f_graph_powerOverTime( m_args.loadgen_out, m_args.power_out )

		#	if( m_args.tmp != "" ):
#		f_parseTemplog( m_args )

def f_graph_powerOverTime( p_loadgen_csv, p_power_csv ):
	global g_power_tz   
	global g_loadgen_tz 
	global g_power_add_td  
	global g_power_sub_td  
	global g_power_window_before_td 
	global g_power_window_after_td  
	
	m_figure_volts = pgo.Figure() # title="Voltage (V) over Time" )
	m_figure_watts = pgo.Figure() # title="Power (W) over Time" )
	m_figure_amps  = pgo.Figure() # title="Current (A) over Time" )
	
	m_loadgen_data = []
	m_power_data = []
	
	m_workname = ""
	m_scenario = ""
	m_testmode = ""
	m_power_state = ""
	m_loadgen_ts = ""
	m_power_ts = ""
	
	# Parse and loadgen data
	try:
		print( f"graph: opening {p_loadgen_csv} for reading..." )
		m_file = open( p_loadgen_csv, 'r' )
	except:
		print( f"graph: error opening file: {p_loadgen_csv}" )
		exit(1)
	
	# loadgen CSV must contain BEGIN and END timestamps
	for m_line in m_file:
		if( re.search("BEGIN", m_line) or re.search("END", m_line) ):
			(m_workname, m_scenario, m_testmode, m_power_state, m_loadgen_ts, m_power_ts) = m_line.split(",", 5)
			m_datetime = datetime.fromisoformat( m_power_ts.strip() )
			m_loadgen_data.append( [m_workname, m_scenario, m_testmode, m_power_state, m_loadgen_ts, m_datetime] )

	# Parse and power data
	try:
		print( f"graph: opening {p_power_csv} for reading..." )
		m_file = open( p_power_csv, 'r' )
#		m_power_data = pandas.read_csv( p_power_csv )
	except:
		print( f"graph: error opening file: {p_power_csv}",  )
		exit(1)
	
	# power CSV	must contain time and power
	# skip first line of headers
	next( m_file )
	for m_line in m_file:
		(m_date, m_time, m_power, m_volt, m_amps) = m_line.split(",")[:5]
		m_datetime = datetime.fromisoformat( m_date + " " + m_time )
		m_power_data.append( [m_datetime, m_power, m_volt, m_amps] )
	
	m_loadgen_iter = iter( m_loadgen_data )
	m_power_iter   = iter( m_power_data )
	for m_loadgen_entry in m_loadgen_iter:
		m_trace_x_time = []
		m_trace_y_watt = []
		m_trace_y_volt = []
		m_trace_y_amps = []
		
		m_power_ts_begin = (m_loadgen_entry[5]).astimezone(g_loadgen_tz)
		m_power_ts_end   = (next(m_loadgen_iter)[5]).astimezone(g_loadgen_tz)

#		print( m_power_ts_begin.strftime("%Y-%m-%D %H:%M:%S.%f")[:-3], "to", m_power_ts_end.strftime("%Y-%m-%D %H:%M:%S.%f")[:-3] )
		
		m_counter = 0
		for m_power_entry in m_power_iter:
		
			#print( m_power_entry[0].strftime("%Y-%m-%D %H:%M:%S.%f")[:-3], m_power_ts_begin.strftime("%Y-%m-%D %H:%M:%S.%f")[:-3], m_power_ts_end.strftime("%Y-%m-%D %H:%M:%S.%f")[:-3] )
			m_power_entry_ts = (m_power_entry[0].replace(tzinfo=g_power_tz)).astimezone(g_loadgen_tz) + g_power_add_td - g_power_sub_td
			
			if( m_power_entry_ts < (m_power_ts_begin - g_power_window_before_td) ):
				continue
			if( m_power_entry_ts > (m_power_ts_end + g_power_window_after_td) ) :
				break
			
			# because of limitations of datetime, offset date by a fixed date
			m_trace_x_time.append( datetime(2011,1,13) + (m_power_entry_ts - m_power_ts_begin) )
			m_trace_y_watt.append( m_power_entry[1] )
			m_trace_y_volt.append( m_power_entry[2] )
			m_trace_y_amps.append( m_power_entry[3] )
			
			m_counter = m_counter + 1
		
		if( m_counter ):
			m_figure_watts.add_trace( pgo.Scatter( x=m_trace_x_time, y=m_trace_y_watt,
												   mode="lines+markers",
												   name=f"{m_loadgen_entry[0]}, {m_loadgen_entry[1]}" ) )
			m_figure_volts.add_trace( pgo.Scatter( x=m_trace_x_time, y=m_trace_y_volt,
												   mode="lines+markers",
												   name=f"{m_loadgen_entry[0]}, {m_loadgen_entry[1]}" ) )
			m_figure_amps.add_trace(  pgo.Scatter( x=m_trace_x_time, y=m_trace_y_amps,
												   mode="lines+markers",
												   name=f"{m_loadgen_entry[0]}, {m_loadgen_entry[1]}" ) )

	m_figure_volts.update_layout( title={'text'   : "Voltage over Time",
	                                     'x'      : 0.5,
										 'y'      : 0.95,
										 'xanchor': 'center',
										 'yanchor': 'top' },
								  xaxis_title="Time (offset between powerlog & loadgen timestamps)",
								  xaxis_tickformat='%H:%M:%S.%L',
								  yaxis_title="Volts (V)" )
	m_figure_watts.update_layout( title={ 'text'  : "Power over Time",
	                                     'x'      : 0.5,
										 'y'      : 0.95,
										 'xanchor': 'center',
										 'yanchor': 'top' },
								  xaxis_title="Time (offset between powerlog & loadgen timestamps)",
								  xaxis_tickformat='%H:%M:%S.%L',
								  yaxis_title="Watts (W)" )
	m_figure_amps.update_layout(  title={'text'   : "Current over Time",
	                                     'x'      : 0.5,
										 'y'      : 0.95,
										 'xanchor': 'center',
										 'yanchor': 'top' },
								  xaxis_title="Time (offset between powerlog & loadgen timestamps)",
								  xaxis_tickformat='%H:%M:%S.%L',
								  yaxis_title="Amps (A)" )
												   
	m_figure_volts.show()
	m_figure_watts.show()
	m_figure_amps.show()
	



# Parse Loadgen log files
# Speciy directory and search for ""*detail.txt" & "*summary.txt"
def f_parseLoadgen( p_dirin, p_fileout ):

	m_workname = [ "resnet50", "resnet",
	               "mobilnet",
				   "gnmt",
				   "ssdmobilenet", "ssd-small",
				   "ssdresnet34",  "ssd-large" ]
	m_metric   = { "offline"      : "Samples per second",
	               "multistream"  : "Samples per query",
				   "singlestream" : "90th percentile latency (ns)",
				   "server"       : "Schehduled Samples per second" }
	m_scenario = ""	
	m_testname = ""
	m_testmode = ""
	m_loadgen_ts = 0
	m_power_ts = ""	
	m_power_state = ""	
	m_score_value = 0
	m_score_valid = ""
	m_counter = 0
	
	m_storage = []
	m_storage.append( ["Workload", "Scenario", "Mode", "State", "Loadgen TS", "System Date", "System Time", "Result", "Score", "Metric"] )

	# Assumes both *detail.txt and *summary.txt files exists
	for m_dirname, m_subdirs, m_filelist in os.walk( p_dirin ):
		for m_filename in m_filelist:
			if m_filename.endswith( 'detail.txt' ):
				m_counter = m_counter + 1
				m_fullpath = os.path.join(m_dirname, m_filename)

				for m_re in m_workname:
					if( re.search( m_re, m_fullpath, re.I ) ):
						m_testname = m_re

				for m_re in m_metric.keys():
					if( re.search( m_re, m_fullpath, re.I ) ):
						m_scenario = m_re
						
				try:
					m_file = open( m_fullpath, 'r' )
				except:
					print( "error opening file:", m_fullpath )
					exit(1)

				for m_line in m_file:
					# Date format is YYYY-MM-DD HH:MM:SS
					if( re.search('time of test', m_line) ):
						m_testmode = ""
						m_power_state = "INIT"
#						m_power_ts = (re.search("(\d*)-(\d*-\d\d).*(\d\d:\d*:\d*)Z$", m_line)).groups()
#						m_power_ts = m_power_ts[1] + "-" + m_power_ts[0] + " " + m_power_ts[2] + ".000"
						m_power_ts = (re.search("(\d*-\d*-\d\d).*(\d\d:\d*:\d*)Z$", m_line)).groups()
						m_power_ts = m_power_ts[0] + " " + m_power_ts[1] + ".000"
					elif( re.search( 'Starting ' + m_testmode + ' mode', m_line) ):
						m_testmode = "START"
						m_power_state = ""
						m_power_ts = ""
					# Date format is MM-DD-YYYY HH:MM:SSS.mmm
					elif( re.search( "POWER_", m_line) ):
						m_power_state = (re.search( "POWER_(\w*)", m_line)).group(1)
						m_power_ts = (re.search('(\d*-\d*)-(\d*)( \d*:\d*:\d*\.\d*)$', m_line)).groups()
						m_power_ts = m_power_ts[1] + "-" + m_power_ts[0] + m_power_ts[2]
					elif( re.search('pid', m_line) and re.search('Scenario', m_line) ):
						m_scenario = (re.search( '(\w*\s?\w*)$', m_line )).group(1)
						m_scenario = (m_scenario.replace( " ", "" )).lower()
						continue
					elif( re.search('Test mode', m_line) ): # and re.search('accuracy', m_line, re.I) ):
						m_testmode = (re.search( "Test mode : (\w*)", m_line)).group(1)
						continue
					else:
						continue

					m_loadgen_ts = (re.search( '(\d*)ns', m_line)).group(1)
					(m_power_ts_date, m_power_ts_time) = m_power_ts.split()
						
					m_storage.append( [m_testname, m_scenario, m_testmode, m_power_state, m_loadgen_ts, m_power_ts_date, m_power_ts_time] )
			
			# Most parameters should be already filled (e.g. testname, scenario, mode)
			elif m_filename.endswith( 'summary.txt' ):
				m_fullpath = os.path.join(m_dirname, m_filename)

				m_score_valid = ""
				m_score_value = ""
				
				try:
					m_file = open( m_fullpath, 'r' )
				except:
					print( "error opening file:", m_fullpath )
					exit(1)

				m_power_state = "DONE"

				for m_line in m_file:
					if( re.search( "Result is", m_line) ):
						m_score_valid = (re.search('Result is : (.*)$', m_line)).group(1)
					elif( re.search( re.escape(m_metric[m_scenario.lower()]), m_line) ):
						m_score_value = (re.search( "(\d*\.?\d*)$", m_line, re.I)).group(1)
#					else:
						# nothing
					continue

				m_storage.append( [m_testname, m_scenario, m_testmode, m_power_state, "", "", "", m_score_valid, m_score_value, m_metric[m_scenario.lower()]] )


	print( "{} loadgen log files found and parsed".format(m_counter) )
	print( "storing CSV data into:", p_fileout )

	try:
		with open( p_fileout, 'w', newline='') as m_file:
			m_csvWriter = csv.writer( m_file, delimiter=',' )

			for m_entry in m_storage:
				m_csvWriter.writerow( m_entry )
		m_file.close()
	except:
		print( "error while creating loadgen log csv output file:", p_fileout )
		exit(1)


# Parse PTDaemon Power Log Filename
# Format should be:
#   Time,MM-DD-YYYY HH:MM:SS.SSS,Watts,DD.DDDDDD,Volts,DDD,DDDDDD,Amps,D.DDDDDD,PF,D.DDDDDD,Mark,String
def f_parsePowerlog( p_filein, p_fileout ):
	m_counter = 0
	m_storage = []

	try:
		m_file = open( p_filein, 'r' )
		print( "opening power log file:", p_filein )
	except:
		print( "error opening power log file:", p_filein )
		exit(1)

	# Create headers
	# Relabel & split date & time for better parsing
	m_line = m_file.readline()
	m_line = m_line.replace( "Time", "Date", 1 )
	m_line = m_line.replace( " ", ",Time,", 1)
	m_storage.append( m_line.split(',')[::2] )

	# Store data
	for m_line in m_file :
		m_counter = m_counter + 1
		m_line = m_line.strip()
		m_line = m_line.replace( "Time", "Date", 1 )
		m_line = m_line.replace( " ", ",Time,", 1)
		m_line = m_line.split(',')[1::2]

		# need to re-order date to iso format
		m_line[0] = m_line[0][-4:] + m_line[0][-5:-4] + m_line[0][:5]
		
		m_storage.append( m_line )

	m_file.close()

	print( "done parsing PTDaemon power log.  {} entries processed".format(m_counter) )
	print( "storing CSV data into:", p_fileout )

	try:
		with open( p_fileout, 'w', newline='') as m_file:
			m_csvWriter = csv.writer( m_file, delimiter=',' )

			for m_entry in m_storage:
				m_csvWriter.writerow( m_entry )
		m_file.close()
	except:
		print( "error while creating PTDaemon power log csv output file:", p_fileout )
		exit(1)



def f_parseParameters():
	m_argparser = argparse.ArgumentParser()

	# Filename options
	# Input
	m_argparser.add_argument( "-lgi", "--loadgen_in",  help="Specify directory of loadgen log files",
	 												   default="" )
	m_argparser.add_argument( "-pli", "--power_in",    help="Specify PTDaemon power log file",
	 												   default="" )

	# Output
	m_argparser.add_argument( "-lgo", "--loadgen_out", help="Specify loadgen CSV output file",
	 												   default="loadgen_out.csv" )
	m_argparser.add_argument( "-plo", "--power_out",   help="Specify power CSV output file",
	 												   default="power_out.csv" )

	# Function options
	m_argparser.add_argument( "-g", "--graph",         help="Draw/output graph of power over time (default input: output loadgen and power CSVs)",
													   action="store_true")
	m_argparser.add_argument( "-s", "--stats",         help="Calculates power stats based on timestamps (both power and loadgen logs required)",
													   action="store_true")

	m_args = m_argparser.parse_args()

	if( m_args.power_in == m_args.power_out ):
		print( "Power log output file cannot be the same as power log input file!")
		exit(1)

	return m_args


if __name__ == '__main__':
	main()
