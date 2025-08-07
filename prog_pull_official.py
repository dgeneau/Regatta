import pandas as pd
import numpy as np
import streamlit as st

st.title('Prog with Official times')


races= pd.read_excel('/Users/danielgeneau/Library/CloudStorage/OneDrive-SharedLibraries-RowingCanadaAviron/HP - Staff - SSSM/General/Performance Analysis/FISA GPS Databases/Sources/races.xlsx')
results = pd.read_excel('/Users/danielgeneau/Library/CloudStorage/OneDrive-SharedLibraries-RowingCanadaAviron/HP - Staff - SSSM/General/Performance Analysis/FISA GPS Databases/Sources/result.xlsx')
current_results = pd.read_excel('/Users/danielgeneau/Library/CloudStorage/OneDrive-SharedLibraries-RowingCanadaAviron/HP - Staff - SSSM/General/Performance Analysis/FISA GPS Databases/Sources/current_results.xlsx')

#filtering race data from only the past 2 years

races = races[races['year']>2022]

_='''
Prognostic Dictionary
'''
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
            "BM8+":"6.269592476",
			"BM4-":"5.899705015",
			"BM2-":"5.376344086",
			"BM4x":"6.024096386",
			"BM2x":"5.555555556",
			"BM1x":"5.115089514",
			"BW8+":"5.66572238",
			"BW4-":"5.347593583",
			"BW2-":"4.901960784",
			"BW4x":"5.464480874",
			"BW2x":"5.037783375",
			"BW1x":"4.672897196",
            "JM8+":"6.269592476",
			"JM4-":"5.899705015",
			"JM2-":"5.376344086",
			"JM4x":"6.024096386",
			"JM2x":"5.555555556",
			"JM1x":"5.115089514",
			"JW8+":"5.66572238",
			"JW4-":"5.347593583",
			"JW2-":"4.901960784",
			"JW4x":"5.464480874",
			"JW2x":"5.037783375",
			"JW1x":"4.672897196",
                }



_='''
Functions for Processing
'''
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




event = st.selectbox('Select Event to Analyze', races['competition'].unique())
event_races = races[races['competition']== event]


days= pd.DataFrame({'datetime': pd.to_datetime(event_races['Date'])})



# Extract date (ignoring time)
event_races['Date (day)'] = days['datetime'].dt.date


day = st.selectbox('Select Day for Analysis', event_races['Date (day)'].unique())


all_day_results = []



day_data = event_races[event_races['Date (day)'] == day]

for BC in day_data['boatClass'].unique():
    # Skip if BC not in prog_dict
    if BC not in prog_dict:
        continue
    prog = prog_dict[BC]

    race_id_frame = day_data[day_data['boatClass'] == BC]
    id_list = race_id_frame['raceId']

    for race_id in id_list:
        race_times = results[results['id'] == race_id]
        race_times = race_times[race_times['d2000m_Rank'] == 1]

        if not race_times.empty:
            
            time_sec = pd.to_numeric(race_times['d2000m_TotalSeconds'].iloc[0], errors='coerce')
            if not pd.isna(time_sec) and prog:
                prognostic = round(((2000 / float(time_sec)) / float(prog))*100,2)
            else:
                prognostic = None

            all_day_results.append({
                'Race Phase': race_times['DisplayName'].iloc[0],
                'Boat Class': BC,
                'Country': race_times['bt_DisplayName'].iloc[0],
                'Official Time': race_times['bt_ResultTime'].iloc[0][3:],
                'Average Split': speed_to_split(2000/float(time_sec)),
                'Race ID': race_id,
                'Speed (m/s)': round(2000/float(time_sec),1),
                'Prognostic': prognostic,
                'Date': day
            })
# Combine everything into one clean DataFrame
day_results = pd.DataFrame(all_day_results)


class_bests = (
    day_results.loc[day_results.groupby("Boat Class")["Prognostic"].idxmax()]
    .sort_values("Prognostic", ascending=False)
    .reset_index(drop=True)
)


class_bests = class_bests.drop('Race ID', axis=1)
class_bests = class_bests.drop('Date', axis=1)

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
    story.append(
        Paragraph(f"{event}", styles["BodyText"])
    )
    story.append(Spacer(1, 12))

    # ── Results table (LongTable → spills automatically) ───────────────────
    data      = [class_bests.columns.tolist()] + class_bests.values.tolist()
    col_w     = [1.5 * inch] + [0.85 * inch] * (len(data[0]) - 1)
    table     = LongTable(
        data, colWidths=col_w, repeatRows=1           # header repeats
    )

    # --- Identify the column that holds the % values --------------------------
    prog_col = class_bests.columns.tolist().index("Prognostic")
    
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
    pdf_data = prog_rep(class_bests, can_row_logo, day)
    st.download_button(
        label="Download Training Report",
        data=pdf_data,
        file_name=f"{day}_Prog_report.pdf",
        mime="application/pdf"
    )
