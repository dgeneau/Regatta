'''
this is dumb I have to do this

'''

import streamlit as st
import pandas as pd
import numpy as np 
import glob
import plotly.graph_objects as go
from scipy.signal import savgol_filter


# adding in prognostic times to look across event types and find the gap

prog_dict = {
			"M8+":"6.269592476",
			"M4-":"5.899705015",
			"M2-":"5.376344086",
			"M4x":"6.024096386",
			"M2x":"5.555555556",
			"M1x":"5.115089514",
			"W8+":"5.66572238",
			"W4-":"5.347593583",
			"W2-":"4.901960784",
			"W4x":"5.464480874",
			"W2x":"5.037783375",
			"W1x":"4.672897196",
                }


def speed_to_split(speed):
    seconds = 500/float(speed)
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"

def sec_to_split(seconds):
    
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"

def rename_duplicate_columns(columns):
    column_counts = {}
    new_columns = []
    
    for column in columns:
        if column in column_counts:
            column_counts[column] += 1
            new_columns.append(f"{column}{column_counts[column]}")
        else:
            column_counts[column] = 0
            new_columns.append(column)
    
    return new_columns



st.set_page_config(layout="wide")



pic, title = st.columns([.1, .9])
with pic: 
	st.image('rowing_canada.png', width = 90)

# Dictionary for mapping the shorthand to readable form
class_mapping = {
    "MSCULL1": "M1x",
    "MSCULL2": "M2x",
    "MSCULL4": "M4x",
    "WSCULL1": "W1x",
    "WSCULL2": "W2x",
    "WSCULL4": "W4x",
    "MNOCOX2": "M2-",
    "MNOCOX4": "M4-",
    "MCOXED8": "M8+",
    "WNOCOX2": "W2-",
    "WNOCOX4": "W4-",
    "WCOXED8": "W8+",
    "XCOXED4": "Mix4-",
    "XSCULL2": "Mix2x"
}

event_mapping = {
    "HEAT": "Heat",
    "REP-": "Repechage",
    "QFNL": "Quarterfinal",
    "SFNL": "Semifinal",
    "FNL-": "Final",
    "PREL": "Preliminary"
}


# Function to clean and transform the data
def clean_data(data):
    cleaned_data = []
    for entry in data:
        # Extracting the class code
        class_code = entry[3:12].replace("-", "")
        
        
        # Checking for para boat class
        if 'PR' in entry:
            class_code += entry[12:16].replace("-", "")
        # Checking for lightweight class
        if '-L-' in entry:
            class_code += 'L'
        
        # Extracting the event type and number
        event_type_code = entry[28:32]
        event_number = entry[32:36].lstrip("0")
        
        # Mapping class code to readable form
        if class_code in class_mapping:
            readable_class = class_mapping[class_code]
        elif 'PR' in class_code:
            readable_class = f"PR{class_code[-2]} {class_mapping[class_code[:-3]]}"
        elif class_code.endswith('L') and class_code[:-1] in class_mapping:
            readable_class = f"L{class_mapping[class_code[:-1]]}"
        else:
            readable_class = class_code
        
        # Mapping event type to readable form
        race_phase = event_mapping.get(event_type_code, event_type_code)
        
        # Creating the readable format
        readable_entry = f"{readable_class} - {race_phase} {event_number}"
        cleaned_data.append(readable_entry)
    
    return cleaned_data

with title:
	st.title('Canada Rowing Regatta Analysis')

file_path  = 'GPS_Data'

events = glob.glob(f'{file_path}/**')
event_list = []
for regatta in events:
	regatta = regatta.split('/')[-1]
	event_list.append(regatta)

event = st.selectbox('Select Event for Analysis', event_list)

race_list = glob.glob(f'{file_path}/{event}/**.csv')

_='''
Hard coding in the prog analysis

- need prog per boat per race
'''
from pathlib import Path
class_type          = []
boat_country_list   = []
prog_list           = []
speed_list          = []
rate_list           = []
stage_type          = []
time_list           = []
split_list          = []

for data_file in race_list:
	# grab the class from the filename first, so we have it even if read_csv fails
	b_class = Path(data_file).stem.split('_')[3]
	stage = Path(data_file).stem.split('_')[4]
	try:
		df = pd.read_csv(data_file, delimiter=';')
		
		prog = prog_dict[b_class]
		
		

		for col in df.columns:
			if 'ShortName' in col:
				boat_country_list.append(df[col].iloc[0])
				class_type.append(b_class)          # ⇦ keep lists in sync
				stage_type.append(stage)
			elif 'Speed' in col:
				prog_list.append(round((df[col].mean() / float(prog)) * 100, 2))
				speed_list.append(round(df[col].mean(), 2))
				#split = speed_to_split(df[col].mean())
				#time_list.append(sec_to_split(2000/df[col].mean()))

			elif 'Stroke' in col:
				rate_list.append(df[col].mean())
				



	except Exception as e:
		#st.write(e)
		continue

#st.write(split_list)
# build the final dataframe
prog_df = pd.DataFrame({
    'Boat'  : boat_country_list,
	'Race Stage' : stage_type,
    'Class' : class_type,
    'Prog'  : prog_list, 
	'Rate' : rate_list,
    'Speed'  : speed_list,
	#'Average Split' : split_list, 
	#'Race Time (GPS)': time_list
})

class_bests = prog_df.groupby('Class').max()
#class_bests['Prog'] = class_bests['Prog'].round(2)
#class_bests['Speed'] = class_bests['Speed'].round(2)

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, LongTable,
    TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet


can_row_logo = "https://images.squarespace-cdn.com/content/v1/5f63780dbc8d16716cca706a/1604523297465-6BAIW9AOVGRI7PARBCH3/rowing-canada-new-logo.jpg"



def prog_rep(class_bests, logo_url):

    """
    Build a multi-page PDF race report whose results table can run
    over any number of pages. The plot is forced onto its own page.
    """


    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=50, rightMargin=50,                                                                           
        topMargin=50,  bottomMargin=50
    )
    styles = getSampleStyleSheet()
    story  = []
    # ── Logo + title row ────────────────────────────────────────────────────
    logo_w, logo_h = 50, 50
    story.append(
        Image(logo_url, width=logo_w, height=logo_h, hAlign="LEFT")
    )
    story.append(
        Paragraph("<b>Daily Prognostic Report</b>", styles["Title"])
    )
    story.append(Spacer(1, 12))

    # ── Results table (LongTable → spills automatically) ───────────────────
    data      = [class_bests.columns.tolist()] + class_bests.values.tolist()
    col_w     = [0.5 * inch] + [0.85 * inch] * (len(data[0]) - 1)
    table     = LongTable(
        data, colWidths=col_w, repeatRows=1           # header repeats
    )

    # --- Identify the column that holds the % values --------------------------
    prog_col = class_bests.columns.tolist().index("Prog")
    
    # --- Base table style -----------------------------------------------------
    style = TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEABOVE", (0, 1), (-1, 1), 1, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
    ])

    # --- Add a text-colour command for every data row -------------------------
    # row 0 is the header, so start at row 1
    for row_idx, row in enumerate(class_bests.itertuples(index=False), start=1):
        # pull the value, strip '%' if it is already a string like "87 %"
        raw = row[prog_col]
        pct = float(str(raw).replace("%", ""))          # → 87.0 for "87 %"

        # map 0 → blue, 100 → red using a simple linear gradient in RGB
        # feel free to tweak the mapping for a different palette
        lo, hi = 70.0, 105.0                        # adjust if the range changes
        if pct <= lo:
            closeness = 0                           # pure blue
        else:
            closeness = min((pct - lo) / (hi - lo), 1)   # 70→0 … 100→1

        colour = colors.Color(red=closeness,
                            green=0,
                            blue=1 - closeness)

        style.add("TEXTCOLOR", (prog_col, row_idx), (prog_col, row_idx), colour)

    # --- Build the table with the augmented style -----------------------------
    table = LongTable(data, colWidths=col_w, repeatRows=1)
    table.setStyle(style)

    story.append(table)
    

    # ── Build PDF ───────────────────────────────────────────────────────────
    doc.build(story)
    buffer.seek(0)
    return buffer
    

_='''
show_prog = False
show_prog = st.checkbox('Show Best Progs')
if st.button("Generate Prog Report"):
    pdf_data = prog_rep(class_bests, can_row_logo)
    st.download_button(
        label="Download Training Report",
        data=pdf_data,
        file_name=f"Prog_report.pdf",
        mime="application/pdf"
    )


if show_prog:
	st.write(class_bests)

'''

race_display = [] 
b_class_list = []
_='''
if 'WCH_2024_1' or 'U23WCH_2024_1' == event: 

	for file in race_list:
		parts = file.split('/')[-1].split('.')[0].split('_')

		race_display.append('_'.join(parts[3:]))
'''

if event == 'ECH_2025_1': 
	for file in race_list: 
		race_display.append(file.split('/')[-1].split('.')[0])

elif event == 'WCP2_2025_1': 
	for file in race_list:
		display = file.split('/')[-1].split('.')[0]
		race_display.append(display)
		b_class = display.split('_')[3]
		b_class_list.append(b_class)
		



else: 
	for file in race_list:
		race_display.append(file.split('/')[-1].split('_')[-1].split('.')[0])


race_display = sorted(race_display)


races = st.multiselect('Select Race(s) for Analysis', race_display)




def convert_seconds_to_time(seconds):
    # Create a timedelta object
    time_delta = pd.to_timedelta(seconds, unit='s')
    
    # Extract minutes, seconds and milliseconds
    minutes = time_delta.components.minutes
    seconds = time_delta.components.seconds
    milliseconds = time_delta.components.milliseconds
    #milliseconds = str(milliseconds)
    #milliseconds = int(milliseconds)


    
    # Format the time as mm:ss.ms
    formatted_time = f"{minutes:02}:{seconds:02}.{milliseconds:01}"
    #formatted_time = f"{minutes:02}:{seconds:02}"
    return formatted_time

def time_to_seconds(time_str):
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds



if len(races)<1: 
	st.header('Select Race for Analysis')
	st.write('Select one or multiple races from the select-box above')
	st.stop()

if races is not None:

	dataframes = []
	selected_b_class = []
	if 'Paris' or 'ECH_2025_1' in event:
		for i, race in enumerate(races):

			sel_b_class = b_class_list[i]
			data = f'{file_path}/{event}/{race}.csv'
			df = pd.read_csv(data, delimiter=';')
			df.columns = [f"{col}_{i+1}" for col in df.columns]
			dataframes.append(df)
			selected_b_class.append(sel_b_class)

	else: 

		for i, race in enumerate(races):


			data = f'{file_path}/{event}/{event}_{race}.csv'
			df = pd.read_csv(data, delimiter=';')
			df.columns = [f"{col}_{i+1}" for col in df.columns]
			dataframes.append(df)

			sel_b_class = b_class_list[i]
			selected_b_class.append(sel_b_class)

	# Concatenate all dataframes horizontally
	df = pd.concat(dataframes, axis=1)

	distance_columns = [col for col in df.columns if 'Distance' in col]
	columns_to_drop = distance_columns[1:]  # Keep the first 'Distance' column and drop the rest

	df.drop(columns=columns_to_drop, inplace=True)
	df.rename(columns={'Distance_1': 'Distance'}, inplace=True)
	
	



	#df = pd.read_csv(data, delimiter=';')

	distance_list = df['Distance']


	col_names = []
	for dis in distance_list:
	    col_names.append(str(dis)+'m')
		
	selected_columns = [col for col in df.columns if 'ShortName' in col]
	phase = data.split('/')[-1].split('.')[0].split('_')[-1]


	speed_columns = [col for col in df.columns if col.startswith('Speed')]
	stroke_columns = [col for col in df.columns if col.startswith('Stroke')]
	name_list  = [col for col in df.columns if col.startswith('ShortName')]
	splits_columns = [col for col in df.columns if col.startswith('Time')]
	times = df[splits_columns].iloc[-1,:]
	times_in_seconds = [(label, time_to_seconds(time)) for label, time in times.items()]

	# Sort the list by the time in seconds
	sorted_times = sorted(times_in_seconds, key=lambda x: x[1])

	rank_dict = {label: rank+1 for rank, (label, _) in enumerate(sorted_times)}

	# Create an array of the ranks based on the original order
	ranks = [rank_dict[label] for label in times.keys()]



	#Plotting Things

	country_list = []
	lane_list = []
	err_list = []
	plain_country = []

	for col in name_list:
		lane = col.split('_')[0][-1]
		lane_list.append(lane)
		plain_country.append(df[col][0])
		

		country_list.append(f'{df[col][0]}, {lane}')
		#country_list.append(df[col][0])




	
	lane_det = st.checkbox('Overall Timing Info')
	#lane_det = False


	if lane_det == True:
		st.header('Timing Summary')
		st.write('Approximated final times as calculated by race GPS Data.')

		#for country in country_list:
		final_times = []
		for lane in range(0, len(country_list)):
			#total_time = convert_seconds_to_time(float(times_in_seconds[country_list.index(country)][1]))
			country = country_list[lane]
			total_time = convert_seconds_to_time(float(times_in_seconds[country_list.index(country)][1]))
			final_times.append(total_time)

		times = pd.DataFrame()
		times['Country'] = plain_country
		times['Lane'] = lane_list
		times['Race Time'] = final_times
		st.dataframe(times.set_index(times.columns[0]), use_container_width = True)
			

	st.header('Graphical Analysis')

	_='''

	lane_filter = st.checkbox('Filter Results by Lane')


	if lane_filter == True: 
		lane_sel = st.selectbox('Select Lanes for Analysis', range(1,len(country_list)+1))
		lane_sel = str(lane_sel)
		countries = [country for country in country_list if lane_sel in country]

		st.write(df)

	'''

	col1, col2  = st.columns([6, 4])

	vel_fig = go.Figure()
	
	for i in range(0,len(speed_columns)):

		
		if df[speed_columns[i]].mean()>.5:
			vel_fig.add_trace(go.Scatter(y=savgol_filter(df[speed_columns[i]],30,2),
										x = df['Distance'],
		                                              mode='lines',
		                                              name=f'{country_list[i]}'))
		else: 
			st.write(f'Error in Data for Country {country_list[i]}')
			err_list.append(i)

	vel_fig.update_layout(
    title='Boat Velocity Vs. Distance',
    xaxis_title='Distance (m)',
    yaxis_title='Velocity (m/s)')

	with col1:
		st.plotly_chart(vel_fig, use_container_width=True)

	#SR
	stroke_fig = go.Figure()
	for i in range(0,len(stroke_columns)):
		try:
			stroke_fig.add_trace(go.Scatter(y=savgol_filter(df[stroke_columns[i]][df[stroke_columns[i]]>20],30,2),
										x = df['Distance'],
		                                              mode='lines',
		                                              name=country_list[i]))
		except:
			pass

	stroke_fig.update_layout(
    title='Boat Stroke Rate Vs. Distance',
    xaxis_title='Distance (m)',
    yaxis_title='Stroke Rate (SPM)')



	with col2:
		st.plotly_chart(stroke_fig, use_container_width=True)


	

	one_index = np.where(df['Distance'] == 250)[0][0]
	two_index = np.where(df['Distance'] == 500)[0][0]
	three_index = np.where(df['Distance'] == 750)[0][0]
	four_index = np.where(df['Distance'] == 1000)[0][0]
	five_index = np.where(df['Distance'] == 1250)[0][0]
	six_index = np.where(df['Distance'] == 1500)[0][0]
	seven_index = np.where(df['Distance'] == 1750)[0][0]
	eight_index = np.where(df['Distance'] == 2000)[0][0]

	avg_vel_250 = df[speed_columns][:one_index].mean()
	avg_vel_500 = df[speed_columns][one_index:two_index].mean()
	avg_vel_750 = df[speed_columns][two_index:three_index].mean()
	avg_vel_1000 = df[speed_columns][three_index:four_index].mean()
	avg_vel_1250 = df[speed_columns][four_index:five_index].mean()
	avg_vel_1500 = df[speed_columns][five_index:six_index].mean()
	avg_vel_1750 = df[speed_columns][six_index:seven_index].mean()
	avg_vel_2000 = df[speed_columns][seven_index:eight_index].mean()
	avg_vel_total = df[speed_columns].mean()

	avg_sr_250 = df[stroke_columns][:one_index].mean()
	avg_sr_500 = df[stroke_columns][one_index:two_index].mean()
	avg_sr_750 = df[stroke_columns][two_index:three_index].mean()
	avg_sr_1000 = df[stroke_columns][three_index:four_index].mean()
	avg_sr_1250 = df[stroke_columns][four_index:five_index].mean()
	avg_sr_1500 = df[stroke_columns][five_index:six_index].mean()
	avg_sr_1750 = df[stroke_columns][six_index:seven_index].mean()
	avg_sr_2000 = df[stroke_columns][seven_index:eight_index].mean()
	avg_sr_total = df[stroke_columns].mean()

	

	data = {
    'Country, Lane': [],
    'Rank': [],
    '250m Split': [],
    '500m Split': [],
    '750m Split': [],
    '1000m Split': [],
    '1250m Split': [],
    '1500m Split': [],
    '1750m Split': [],
    '2000m Split': [], 
    '250m Stroke': [],
    '500m Stroke': [],
    '750m Stroke': [],
    '1000m Stroke': [],
    '1250m Stroke': [],
    '1500m Stroke': [],
    '1750m Stroke': [],
    '2000m Stroke': [],
    '250m Speed': [],
    '500m Speed': [],
    '750m Speed': [],
    '1000m Speed': [],
    '1250m Speed': [],
    '1500m Speed': [],
    '1750m Speed': [],
    '2000m Speed': [],

}
	
	
	for i in range(len(avg_vel_250)):
		try:
		
			data['Country, Lane'].append(country_list[i])
			data['Rank'].append(ranks[i])
			data['250m Split'].append(convert_seconds_to_time(500 / avg_vel_250[i]))
			data['500m Split'].append(convert_seconds_to_time(500 / avg_vel_500[i]))
			data['750m Split'].append(convert_seconds_to_time(500 / avg_vel_750[i]))
			data['1000m Split'].append(convert_seconds_to_time(500 / avg_vel_1000[i]))
			data['1250m Split'].append(convert_seconds_to_time(500 / avg_vel_1250[i]))
			data['1500m Split'].append(convert_seconds_to_time(500 / avg_vel_1500[i]))
			data['1750m Split'].append(convert_seconds_to_time(500 / avg_vel_1750[i]))
			data['2000m Split'].append(convert_seconds_to_time(500 / avg_vel_2000[i]))
			data['250m Stroke'].append(round(avg_sr_250[i],2))
			data['500m Stroke'].append(round(avg_sr_500[i],2))
			data['750m Stroke'].append(round(avg_sr_750[i],2))
			data['1000m Stroke'].append(round(avg_sr_1000[i],2))
			data['1250m Stroke'].append(round(avg_sr_1250[i],2))
			data['1500m Stroke'].append(round(avg_sr_1500[i],2))
			data['1750m Stroke'].append(round(avg_sr_1750[i],2))
			data['2000m Stroke'].append(round(avg_sr_2000[i],2))
			data['250m Speed'].append(round(avg_vel_250[i], 2))
			data['500m Speed'].append(round(avg_vel_500[i], 2))
			data['750m Speed'].append(round(avg_vel_750[i], 2))
			data['1000m Speed'].append(round(avg_vel_1000[i], 2))
			data['1250m Speed'].append(round(avg_vel_1250[i], 2))
			data['1500m Speed'].append(round(avg_vel_1500[i], 2))
			data['1750m Speed'].append(round(avg_vel_1750[i], 2))
			data['2000m Speed'].append(round(avg_vel_2000[i], 2))

		except:
			pass


	for err in err_list:	
		data['Country, Lane'].pop(err)
		data['Rank'].pop(err)	
	
	splits_unsorted = pd.DataFrame(data)
	splits = splits_unsorted.sort_values(by = 'Rank').reset_index(drop=True)


	
	#highlight_row = splits[splits['Country, Lane'].str.contains('CAN')].index[0] if 'CAN' in splits['Country, Lane'].values else None
	highlight_row = splits[splits['Country, Lane'].str.contains('CAN')].index[0] if splits['Country, Lane'].str.contains('CAN').any() else None

	# Define colors for the cells
	fill_colors = [['aliceblue' for _ in range(len(splits))] for _ in splits.columns]

	# If 'CAN' exists, highlight the corresponding row
	if highlight_row is not None:
	    for i in range(len(splits.columns)):
	        fill_colors[i][highlight_row] = 'palevioletred'



	splits_plot = go.Figure()

	transposed_split = splits_unsorted.T
	transposed_split.columns = transposed_split.iloc[0,:]
	transposed_split = transposed_split.iloc[2:, :]
	transposed_split.columns = rename_duplicate_columns(transposed_split.columns)


	
	for col in transposed_split.columns:
		splits_plot.add_trace(go.Scatter(y=pd.to_datetime(transposed_split[col]), 
			x = [250, 500, 750, 1000, 1250, 1500, 1750, 2000], 
			name = col))

	splits_plot.update_layout(
    title='Race Split Vs. Distance',
    xaxis_title='Distance (m)',
    yaxis=dict(
        title='Time for 500m',
        autorange='reversed',  # Invert the y-axis
    ))
		

	st.plotly_chart(splits_plot, use_container_width = True)

	first_two_cols = splits.iloc[:, :2]
	remaining_cols = splits.iloc[:, 2:]
	distance_columns = ['250m', '500m', '750m', '1000m', '1250m', '1500m', '1750m', '2000m']


	# Create a new DataFrame to hold the concatenated values
	concatenated_splits = pd.DataFrame(first_two_cols)

	# Iterate through the columns in steps of 8
	for j in range(8):
	    new_col = {}
	    col_1 = remaining_cols.columns[j] if j < remaining_cols.shape[1] else None
	    col_9 = remaining_cols.columns[j+8] if (j+8) < remaining_cols.shape[1] else None
	    col_17 = remaining_cols.columns[j+16] if (j+16) < remaining_cols.shape[1] else None
	    
	    concatenated_values = remaining_cols.apply(lambda row: '<br>'.join([str(row[col]) for col in [col_1, col_9, col_17] if col is not None]), axis=1)
	    
	    if j < len(distance_columns):
	        new_col[distance_columns[j]] = concatenated_values
	    
	    concatenated_splits = pd.concat([concatenated_splits, pd.DataFrame(new_col)], axis=1)


	st.header('Race Split Breakdown')
	st.write('Data provided by country in the order of split, stroke rate, average velocity for each 250m section')
	#concatenated_splits = concatenated_splits.iloc[:, :-8]
	splits_fig  = go.Figure(data=[go.Table(
	    header=dict(values=list(concatenated_splits.columns),
	                fill_color='grey',
	                font=dict(size=16, color='white'),
	                align='left'),
	    cells=dict(values=[concatenated_splits[col] for col in concatenated_splits.columns],
	               fill=dict(color=fill_colors),
	               font=dict(size=14, color='black'),
	               align='left'))
	])
	splits_fig.update_layout(height=800) 
	st.plotly_chart(splits_fig, use_container_width=True)










