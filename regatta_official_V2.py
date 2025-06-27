import streamlit as st
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# prognostic times for reference (if you need them later)
prog_dict = {
    "M8+":"6.269592476","M4-":"5.899705015","M2-":"5.376344086",
    "M4x":"6.024096386","M2x":"5.555555556","M1x":"5.115089514",
    "W8+":"5.66572238","W4-":"5.347593583","W2-":"4.901960784",
    "W4x":"5.464480874","W2x":"5.037783375","W1x":"4.672897196",
}

def convert_seconds_to_time(seconds):
    td = pd.to_timedelta(seconds, unit='s')
    m = td.components.minutes
    s = td.components.seconds
    ms = td.components.milliseconds // 100  # one digit
    return f"{m:02d}:{s:02d}.{ms:01d}"

def time_to_seconds(time_str):
    m, s = map(float, time_str.split(':'))
    return m*60 + s

def rename_duplicate_columns(cols):
    counts = {}
    out = []
    for c in cols:
        if c in counts:
            counts[c] += 1
            out.append(f"{c}{counts[c]}")
        else:
            counts[c] = 0
            out.append(c)
    return out

st.set_page_config(layout="wide")
pic, title = st.columns([0.1, 0.9])
with pic:
    st.image('rowing_canada.png', width=90)
with title:
    st.title("Canada Rowing Regatta Analysis")



# --- 1) Select Event and build a list of all CSVs ---
DATA_DIR = "GPS_Data"
events = [Path(d).name for d in glob.glob(f"{DATA_DIR}/*") if Path(d).is_dir()]
event = st.selectbox("Select Event for Analysis", events)

all_csvs = glob.glob(f"{DATA_DIR}/{event}/**/*.csv", recursive=True)
race_info = []
for path in all_csvs:
    p = Path(path)
    display = p.stem  # e.g. "WCH_2024_1_XXX_Heat_01"
    race_info.append({
        "display": display,
        "file_path": str(p)
    })

race_display = sorted(r["display"] for r in race_info)
# --- 2) Let user pick one or more races ---
races = st.multiselect("Select Race(s) for Analysis", race_display)


if not races:
    st.header("Select Race for Analysis")
    st.write("Select one or multiple races from the select-box above")
    st.stop()

# --- 3) Load all selected CSVs, rename their columns uniquely, then concat ---
dataframes = []
for idx, race_name in enumerate(races, start=1):
    meta = next(r for r in race_info if r["display"] == race_name)
    df_r = pd.read_csv(meta["file_path"], delimiter=';')
    # suffix every column so they don't collide when we concat
    df_r.columns = [f"{c}_{idx}" for c in df_r.columns]
    dataframes.append(df_r)

# combine side-by-side
df = pd.concat(dataframes, axis=1)


# drop extra Distance columns, keep the first
dist_cols = [c for c in df.columns if c.startswith("Distance")]
for c in dist_cols[1:]:
    df.drop(columns=[c], inplace=True)
df.rename(columns={dist_cols[0]: "Distance"}, inplace=True)

# --- 4) Derive the lists of interest from the concatenated df ---
speed_columns  = [c for c in df.columns if c.startswith("Speed")]
stroke_columns = [c for c in df.columns if c.startswith("Stroke")]
name_columns   = [c for c in df.columns if c.startswith("ShortName")]
split_columns  = [c for c in df.columns if c.startswith("Time")]

# compute final times & ranking
final_times = df[split_columns].iloc[-1]
times_secs  = {lbl: time_to_seconds(val) for lbl, val in final_times.items()}
sorted_times = sorted(times_secs.items(), key=lambda x: x[1])
rank_map    = {lbl: rank+1 for rank, (lbl, _) in enumerate(sorted_times)}
ranks       = [rank_map[lbl] for lbl in final_times.index]

# build country & lane lists
country_list = []
lane_list    = []
for col in name_columns:
    lane = col.split("_")[-1]
    country = df[col].iloc[0]
    lane_list.append(lane)
    country_list.append(f"{country}, {lane}")

# --- 5) Optionally show timing summary ---
if st.checkbox("Overall Timing Info"):
    st.header("Timing Summary")
    summary = pd.DataFrame({
        "Country": [c.split(",")[0] for c in country_list],
        "Lane":    lane_list,
        "Race Time": [convert_seconds_to_time(times_secs[lbl]) 
                      for lbl in final_times.index]
    })
    st.dataframe(summary.set_index("Country"), use_container_width=True)

# --- 6) Graphical Analysis ---
st.header("Graphical Analysis")
col1, col2 = st.columns([6, 4])

# Velocity vs Distance
vel_fig = go.Figure()
for col in speed_columns:
    if df[col].mean() > 0.5:
        vel_fig.add_trace(go.Scatter(
            x=df["Distance"],
            y=savgol_filter(df[col], 30, 2),
            mode="lines",
            name=col
        ))
    else:
        st.write(f"Warning: low-mean data in {col}")
with col1:
    st.plotly_chart(vel_fig, use_container_width=True)

# Stroke Rate vs Distance
sr_fig = go.Figure()
for col in stroke_columns:
    try:
        y = savgol_filter(df[col][df[col] > 20], 30, 2)
        sr_fig.add_trace(go.Scatter(
            x=df["Distance"],
            y=y,
            mode="lines",
            name=col
        ))
    except:
        pass
with col2:
    st.plotly_chart(sr_fig, use_container_width=True)

# --- 7) Compute splits & strokes & speeds for each 250m segment ---
breaks = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
idxs   = [np.where(df["Distance"] == b)[0][0] for b in breaks]

avg_vel_sections = []
avg_sr_sections  = []
prev = 0
for cut in idxs:
    avg_vel_sections.append(df[speed_columns][prev:cut].mean())
    avg_sr_sections.append(df[stroke_columns][prev:cut].mean())
    prev = cut

# --- 8) Build a summary table of splits / stroke / speed ---
data = {
    "Country, Lane": country_list,
    "Rank": ranks
}
for i, b in enumerate(breaks):
    # split time for a 500m piece = 500 / avg velocity
    data[f"{b}m Split"]  = [convert_seconds_to_time(500 / v) for v in avg_vel_sections[i]]
    data[f"{b}m Stroke"] = [round(s, 2)                 for s in avg_sr_sections[i]]
    data[f"{b}m Speed"]  = [round(v, 2)                 for v in avg_vel_sections[i]]

splits_unsorted = pd.DataFrame(data)
splits = splits_unsorted.sort_values("Rank").reset_index(drop=True)

# highlight Canada row if present
highlight = None
for i, val in enumerate(splits["Country, Lane"]):
    if "CAN" in val:
        highlight = i
        break

fill_colors = [["white"]*splits.shape[0] for _ in splits.columns]
if highlight is not None:
    for col in range(len(splits.columns)):
        fill_colors[col][highlight] = "lightcoral"

# --- 9) Plot cumulative splits over distance ---
trans = splits_unsorted.T
trans.columns = trans.iloc[0]
trans = trans.iloc[2:, :]  # drop the Country/Lane & Rank rows
trans.columns = rename_duplicate_columns(trans.columns)

split_vs_dist = go.Figure()
for col in trans.columns:
    split_vs_dist.add_trace(go.Scatter(
        x=breaks,
        y=pd.to_datetime(trans[col]),
        name=col
    ))
split_vs_dist.update_layout(
    title="Race Split Vs. Distance",
    xaxis_title="Distance (m)",
    yaxis=dict(title="Time for 500m", autorange="reversed"),
)
st.plotly_chart(split_vs_dist, use_container_width=True)

# --- 10) Show final breakdown table ---
st.header("Race Split Breakdown")
st.write("Split / Stroke / Speed for each 250m by country & lane")
table_fig = go.Figure(data=[go.Table(
    header=dict(values=list(splits.columns), fill_color="grey", font=dict(color="white")),
    cells=dict(values=[splits[col] for col in splits.columns],
               fill_color=fill_colors),
)])
table_fig.update_layout(height=600)
st.plotly_chart(table_fig, use_container_width=True)

st.stop()