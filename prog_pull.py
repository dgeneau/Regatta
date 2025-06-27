
_='''
Hard coding in the prog analysis

- need prog per boat per race
'''
import streamlit as st
import pandas as pd
import numpy as np 
from pathlib import Path
import glob


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
    try:
        seconds = 500/float(speed)
        # Calculate the minutes, seconds, and milliseconds
        minutes = int(seconds // 60)
        seconds_remainder = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
    except: 
         minutes = 0
         seconds_remainder = 0
         milliseconds = 0
    
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
	st.title('Prognostic Analysis')

file_path  = 'GPS_Data'

events = glob.glob(f'{file_path}/**')
event_list = []
for regatta in events:
	regatta = regatta.split('/')[-1]
	event_list.append(regatta)

event = st.selectbox('Select Event for Analysis', event_list)

day_list = glob.glob(f'{file_path}/{event}/*')
day = st.selectbox('Select Event for Analysis', day_list)

race_day = day.split('/')[-1].split('.')[0]
st.write(race_day)

race_list = glob.glob(f'{day}/**.csv')



class_type        = []
boat_country_list = []
prog_list         = []
speed_list        = []
rate_list         = []
stage_type        = []
time_list         = []
split_list        = []

for data_file in race_list:
    b_class = Path(data_file).stem.split('_')[3]
    stage   = Path(data_file).stem.split('_')[4]
    
    try:
        df = pd.read_csv(data_file, delimiter=';')
        prog = prog_dict[b_class]

        # Detect how many boats there are by checking suffix numbers
        suffixes = sorted(set(col[-1] for col in df.columns if col[-1].isdigit()))
        
        for num in suffixes:
            try:
                country = df[f'ShortName{num}'].iloc[0]
                speed = df[f'Speed{num}'].mean()
                rate = df[f'Stroke{num}'].mean()
                prog_val = round((speed / float(prog)) * 100, 2)
                split = speed_to_split(speed)
                time = sec_to_split(2000 / speed)

                # Append all values in sync
                boat_country_list.append(country)
                speed_list.append(round(speed, 2))
                rate_list.append(rate)
                prog_list.append(prog_val)
                split_list.append(split)
                time_list.append(time)
                class_type.append(b_class)
                stage_type.append(stage)
            except KeyError as ke:
                st.write(f"Missing column for num={num}: {ke}")
            except Exception as inner_e:
                st.write(f"Error processing boat {num}: {inner_e}")

    except Exception as e:
        st.write(f"Error reading {data_file}: {e}")
        continue

# build the final dataframe
prog_df = pd.DataFrame({
    'Boat'  : boat_country_list,
	'Race Stage' : stage_type,
    'Class' : class_type,
    'Prog'  : prog_list, 
	'Rate' : rate_list,
    'Speed'  : speed_list,
	'Average Split' : split_list, 
	'Race Time (GPS)': time_list
})

class_bests = (
    prog_df.loc[prog_df.groupby("Class")["Prog"].idxmax()]
    .sort_values("Prog", ascending=False)
    .reset_index(drop=True)
)
st.write(class_bests)




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



def prog_rep(class_bests, logo_url,race_date):

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
        Paragraph(f"<b>Daily Prognostic Report: {race_date}</b>", styles["Title"])
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
    
if st.button("Generate Prog Report"):
    pdf_data = prog_rep(class_bests, can_row_logo, race_day)
    st.download_button(
        label="Download Training Report",
        data=pdf_data,
        file_name=f"Prog_report.pdf",
        mime="application/pdf"
    )


