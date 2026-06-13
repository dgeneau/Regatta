"""Prognostic analysis for every boat in selected races."""

import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle


PROG_SPEEDS = {
    "M8+": 6.269592476,
    "M4-": 5.899705015,
    "M2-": 5.376344086,
    "M4x": 6.024096386,
    "M2x": 5.555555556,
    "M1x": 5.115089514,
    "W8+": 5.66572238,
    "W4-": 5.347593583,
    "W2-": 4.901960784,
    "W4x": 5.464480874,
    "W2x": 5.037783375,
    "W1x": 4.672897196,
}

DATA_ROOT = Path("GPS_Data")


def speed_to_split(speed):
    if pd.isna(speed) or speed <= 0:
        return ""
    seconds = 500 / float(speed)
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds % 60:06.3f}"


def seconds_to_race_time(seconds):
    if pd.isna(seconds) or seconds <= 0:
        return ""
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds % 60:06.3f}"


def prognostic_color(value):
    """Map prognostic percentage to a readable blue-to-red text color."""
    try:
        percentage = float(value)
    except (TypeError, ValueError):
        return colors.black

    closeness = min(max((percentage - 85) / 15, 0), 1)
    return colors.Color(closeness, 0.12, 1 - closeness)


def prognostic_text_style(value):
    colour = prognostic_color(value)
    red, green, blue = (round(channel * 255) for channel in colour.rgb())
    return f"color: rgb({red}, {green}, {blue}); font-weight: 700"


def trend_text_style(value):
    """Color negative time changes green and positive changes red."""
    try:
        change = float(value)
    except (TypeError, ValueError):
        return ""
    if change < -0.05:
        return "color: #15803D; font-weight: 700"
    if change > 0.05:
        return "color: #B91C1C; font-weight: 700"
    return "color: #4B5563; font-weight: 700"


def normalize_boat_identity(boat):
    """Normalize casing while preserving numbered boats as distinct entries."""
    return str(boat).strip().upper()


def normalize_boat_class(boat_class):
    """Map U23, junior, and lightweight classes to the open-class speed."""
    normalized = boat_class
    if len(normalized) > 1 and normalized[0] in {"B", "J"}:
        normalized = normalized[1:]
    if len(normalized) > 1 and normalized[0] == "L":
        normalized = normalized[1:]
    return normalized


def parse_race_metadata(data_file, event):
    stem = Path(data_file).stem
    race_name = stem

    prefix = f"{event}_"
    if stem.startswith(prefix):
        race_name = stem[len(prefix):]

    parts = race_name.split("_")
    boat_class = parts[0] if parts else ""
    stage = " ".join(parts[1:]) if len(parts) > 1 else race_name
    return boat_class, stage, race_name


def boat_suffixes(columns):
    suffixes = []
    for column in columns:
        match = re.fullmatch(r"ShortName(\d+)", column)
        if match:
            suffixes.append(int(match.group(1)))
    return sorted(suffixes)


def analyze_race(data_file, event, group):
    boat_class, stage, race_name = parse_race_metadata(data_file, event)
    prog_class = normalize_boat_class(boat_class)
    prog_speed = PROG_SPEEDS.get(prog_class)
    if prog_speed is None:
        return pd.DataFrame(), f"{race_name}: no prognostic speed for {boat_class}"

    race_df = pd.read_csv(data_file, delimiter=";")
    rows = []

    for suffix in boat_suffixes(race_df.columns):
        required = [f"ShortName{suffix}", f"Speed{suffix}", f"Stroke{suffix}"]
        if any(column not in race_df.columns for column in required):
            continue

        boat = race_df[f"ShortName{suffix}"].dropna()
        speed = pd.to_numeric(race_df[f"Speed{suffix}"], errors="coerce").mean()
        rate = pd.to_numeric(race_df[f"Stroke{suffix}"], errors="coerce")
        rate = rate[rate > 0].mean()

        if boat.empty or pd.isna(speed) or speed <= 0:
            continue

        rows.append(
            {
                "Event": event,
                "Day / Group": group,
                "Race": race_name,
                "Race Stage": stage,
                "Class": boat_class,
                "Boat": str(boat.iloc[0]),
                "Lane": suffix,
                "Prog": round((speed / prog_speed) * 100, 2),
                "Rate": round(rate, 1) if not pd.isna(rate) else None,
                "Speed": round(speed, 2),
                "Average Split": speed_to_split(speed),
                "Race Time (GPS)": seconds_to_race_time(2000 / speed),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["GPS Rank"] = (
            result["Speed"].rank(method="min", ascending=False).astype(int)
        )
        column_order = [
            "Event",
            "Day / Group",
            "Race",
            "Race Stage",
            "Class",
            "GPS Rank",
            "Boat",
            "Lane",
            "Prog",
            "Rate",
            "Speed",
            "Average Split",
            "Race Time (GPS)",
        ]
        result = result[column_order].sort_values(["GPS Rank", "Lane"])

    return result, None


def build_race_catalog():
    records = []
    for data_file in sorted(DATA_ROOT.rglob("*.csv")):
        relative_path = data_file.relative_to(DATA_ROOT)
        path_parts = relative_path.parts

        # Older data is stored under GPS_Data/2024/<event>/...
        if path_parts[0] == "2024" and len(path_parts) >= 3:
            event = path_parts[1]
            group_parts = path_parts[2:-1]
        else:
            event = path_parts[0]
            group_parts = path_parts[1:-1]

        group = " / ".join(group_parts) if group_parts else "."
        _, _, race_name = parse_race_metadata(data_file, event)
        label = f"{event} | {group} | {race_name}"
        records.append(
            {
                "label": label,
                "event": event,
                "group": group,
                "race": race_name,
                "path": data_file,
            }
        )
    return records


def build_matched_boat_trends(all_boats, selected_events):
    trend_source = all_boats.copy()
    trend_source["Boat Identity"] = trend_source["Boat"].map(normalize_boat_identity)
    trend_source["GPS Seconds"] = 2000 / trend_source["Speed"]

    # Use each exact boat's fastest selected race at each event.
    best_indices = trend_source.groupby(["Event", "Boat Identity", "Class"])["Prog"].idxmax()
    event_bests = trend_source.loc[best_indices].copy()

    event_counts = event_bests.groupby(["Boat Identity", "Class"])["Event"].nunique()
    matched_keys = event_counts[event_counts == len(selected_events)].index
    if matched_keys.empty:
        return pd.DataFrame(), pd.DataFrame()

    matched = event_bests.set_index(["Boat Identity", "Class"]).loc[matched_keys].reset_index()
    event_order = {event: position for position, event in enumerate(selected_events)}
    matched["Event Order"] = matched["Event"].map(event_order)
    matched = matched.sort_values(["Boat Identity", "Class", "Event Order"])

    summary_rows = []
    for (boat_identity, boat_class), group in matched.groupby(["Boat Identity", "Class"], sort=True):
        group = group.sort_values("Event Order")
        first = group.iloc[0]
        last = group.iloc[-1]
        time_change = round(last["GPS Seconds"] - first["GPS Seconds"], 2)
        prog_change = round(last["Prog"] - first["Prog"], 2)
        row = {
            "Boat": boat_identity,
            "Class": boat_class,
            "Time Change (s)": time_change,
            "Prog Change": prog_change,
            "Trend": "Faster" if time_change < -0.05 else "Slower" if time_change > 0.05 else "No change",
        }
        for event in selected_events:
            event_row = group[group["Event"] == event].iloc[0]
            row[f"{event} Time"] = event_row["Race Time (GPS)"]
            row[f"{event} Prog"] = event_row["Prog"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    leading_columns = ["Boat", "Class"]
    event_columns = [
        column
        for event in selected_events
        for column in [f"{event} Time", f"{event} Prog"]
    ]
    trailing_columns = ["Time Change (s)", "Prog Change", "Trend"]
    summary = summary[leading_columns + event_columns + trailing_columns]

    detail_columns = [
        "Boat Identity",
        "Class",
        "Event",
        "Race",
        "Race Stage",
        "Prog",
        "Average Split",
        "Race Time (GPS)",
        "GPS Seconds",
    ]
    detail = matched[detail_columns].copy()
    detail["GPS Seconds"] = detail["GPS Seconds"].round(2)
    return summary, detail


def base_pdf_table_style():
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343A40")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A7ADB3")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
        ]
    )


def build_matched_pdf(summary, detail, selected_events):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Matched Boats Trend Report</b>", styles["Title"]),
        Paragraph(f"Events: {', '.join(selected_events)}", styles["BodyText"]),
        Paragraph(
            "Each exact boat and class is represented by its fastest selected race at each event. "
            "Negative time change indicates improvement.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    summary_pdf = summary.copy()
    top_header = ["Boat", "Class"]
    sub_header = ["", ""]
    for event in selected_events:
        top_header.extend([event, ""])
        sub_header.extend(["Time", "Prog"])
    top_header.extend(["Change", "", ""])
    sub_header.extend(["Time (s)", "Prog", "Trend"])

    summary_data = [top_header, sub_header] + summary_pdf.fillna("").values.tolist()
    summary_widths = [0.55 * inch, 0.5 * inch]
    summary_widths += [0.8 * inch, 0.55 * inch] * len(selected_events)
    summary_widths += [0.7 * inch, 0.65 * inch, 0.65 * inch]
    summary_table = LongTable(summary_data, colWidths=summary_widths, repeatRows=2)
    summary_style = base_pdf_table_style()
    summary_style.add("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#343A40"))
    summary_style.add("TEXTCOLOR", (0, 1), (-1, 1), colors.white)
    summary_style.add("SPAN", (0, 0), (0, 1))
    summary_style.add("SPAN", (1, 0), (1, 1))

    event_start_column = 2
    for event_number in range(len(selected_events)):
        start_column = event_start_column + (event_number * 2)
        summary_style.add("SPAN", (start_column, 0), (start_column + 1, 0))

    change_start_column = event_start_column + (len(selected_events) * 2)
    summary_style.add("SPAN", (change_start_column, 0), (change_start_column + 2, 0))

    prog_columns = [
        index for index, column in enumerate(summary_pdf.columns) if column.endswith(" Prog")
    ]
    time_change_column = summary_pdf.columns.get_loc("Time Change (s)")
    trend_column = summary_pdf.columns.get_loc("Trend")
    for row_number, row in enumerate(summary_pdf.itertuples(index=False), start=2):
        for prog_column in prog_columns:
            summary_style.add(
                "TEXTCOLOR",
                (prog_column, row_number),
                (prog_column, row_number),
                prognostic_color(row[prog_column]),
            )
            summary_style.add("FONTNAME", (prog_column, row_number), (prog_column, row_number), "Helvetica-Bold")

        time_change = float(row[time_change_column])
        trend_colour = colors.HexColor("#15803D") if time_change < -0.05 else colors.HexColor("#B91C1C") if time_change > 0.05 else colors.HexColor("#4B5563")
        summary_style.add("TEXTCOLOR", (time_change_column, row_number), (trend_column, row_number), trend_colour)
        summary_style.add("FONTNAME", (time_change_column, row_number), (trend_column, row_number), "Helvetica-Bold")

    summary_table.setStyle(summary_style)
    story.append(summary_table)
    story.extend([PageBreak(), Paragraph("<b>Matched Boat Race Details</b>", styles["Heading1"]), Spacer(1, 8)])

    detail_pdf = detail.drop(columns=["Race Stage", "GPS Seconds"]).rename(
        columns={
            "Boat Identity": "Boat",
            "Average Split": "Avg Split",
            "Race Time (GPS)": "GPS Time",
        }
    )
    detail_data = [detail_pdf.columns.tolist()] + detail_pdf.fillna("").values.tolist()
    detail_widths = [0.65 * inch, 0.55 * inch, 1.15 * inch, 1.7 * inch, 0.55 * inch, 0.8 * inch, 0.85 * inch]
    detail_table = LongTable(detail_data, colWidths=detail_widths, repeatRows=1)
    detail_style = base_pdf_table_style()
    detail_prog_column = detail_pdf.columns.get_loc("Prog")
    for row_number, row in enumerate(detail_pdf.itertuples(index=False), start=1):
        detail_style.add(
            "TEXTCOLOR",
            (detail_prog_column, row_number),
            (detail_prog_column, row_number),
            prognostic_color(row[detail_prog_column]),
        )
        detail_style.add("FONTNAME", (detail_prog_column, row_number), (detail_prog_column, row_number), "Helvetica-Bold")

    detail_table.setStyle(detail_style)
    story.append(detail_table)
    doc.build(story)
    buffer.seek(0)
    return buffer


def build_pdf(results, selected_events):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>All Boats Prognostic Report</b>", styles["Title"]),
        Paragraph(f"Events: {', '.join(selected_events)}", styles["BodyText"]),
        Spacer(1, 12),
    ]

    report_columns = [
        "Event",
        "Day / Group",
        "Race",
        "Class",
        "GPS Rank",
        "Boat",
        "Lane",
        "Prog",
        "Rate",
        "Average Split",
        "Race Time (GPS)",
    ]
    report_df = results.loc[:, report_columns].copy()
    report_df = report_df.rename(
        columns={
            "Day / Group": "Day",
            "GPS Rank": "Rank",
            "Average Split": "Avg Split",
            "Race Time (GPS)": "GPS Time",
        }
    )
    data = [report_df.columns.tolist()] + report_df.fillna("").values.tolist()
    widths = [
        1.15 * inch,
        0.65 * inch,
        1.35 * inch,
        0.55 * inch,
        0.45 * inch,
        0.55 * inch,
        0.4 * inch,
        0.55 * inch,
        0.5 * inch,
        0.8 * inch,
        0.85 * inch,
    ]
    table = LongTable(data, colWidths=widths, repeatRows=1)
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343A40")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A7ADB3")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
        ]
    )

    prog_column = report_df.columns.get_loc("Prog")
    event_column = report_df.columns.get_loc("Event")
    race_column = report_df.columns.get_loc("Race")
    previous_race = None
    for row_number, row in enumerate(report_df.itertuples(index=False), start=1):
        style.add("TEXTCOLOR", (prog_column, row_number), (prog_column, row_number), prognostic_color(row[prog_column]))
        style.add("FONTNAME", (prog_column, row_number), (prog_column, row_number), "Helvetica-Bold")

        race_key = (row[event_column], row[race_column])
        if previous_race is not None and race_key != previous_race:
            style.add("LINEABOVE", (0, row_number), (-1, row_number), 1, colors.HexColor("#4B5563"))
        previous_race = race_key

    table.setStyle(style)
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer


st.set_page_config(layout="wide")
st.title("All Boats Prognostic Analysis")
st.caption("Select races, then compare every boat using the same GPS prognostic calculation.")

race_catalog = build_race_catalog()
event_names = sorted({record["event"] for record in race_catalog})

if not event_names:
    st.error(f"No race files found in {DATA_ROOT}.")
    st.stop()

selected_events = st.multiselect(
    "Select Event(s) for Analysis",
    event_names,
    placeholder="Choose events to compare",
)

if not selected_events:
    st.info("Select one or more events to see their available races.")
    st.stop()

filtered_catalog = [
    record for record in race_catalog if record["event"] in selected_events
]
race_options = {record["label"]: record for record in filtered_catalog}
selected_races = st.multiselect(
    "Select Race(s) Across Events",
    list(race_options),
    placeholder="Optional: choose races, or leave empty to compare all matching boats",
)

automatic_all_races = not selected_races
if automatic_all_races:
    selected_races = list(race_options)
    st.info(
        f"No specific races selected. Comparing matching boats across all "
        f"{len(selected_races)} races in the selected events."
    )

race_results = []
warnings = []
for selected_race in selected_races:
    race_record = race_options[selected_race]
    result, warning = analyze_race(
        race_record["path"],
        race_record["event"],
        race_record["group"],
    )
    if not result.empty:
        race_results.append(result)
    if warning:
        warnings.append(warning)

for warning in warnings:
    st.warning(warning)

if not race_results:
    st.error("No boats with supported prognostic classes were found in the selected races.")
    st.stop()

all_boats = pd.concat(race_results, ignore_index=True)
all_boats = all_boats.sort_values(
    ["Event", "Day / Group", "Race", "GPS Rank", "Lane"]
).reset_index(drop=True)

race_group_columns = ["Event", "Day / Group", "Race", "Class"]
race_bests = all_boats.loc[
    all_boats.groupby(race_group_columns)["Prog"].idxmax(),
    race_group_columns + ["Boat", "Prog", "Average Split", "Race Time (GPS)"],
].rename(
    columns={
        "Boat": "Best Boat",
        "Prog": "Best Prog",
        "Average Split": "Best Average Split",
        "Race Time (GPS)": "Best Race Time (GPS)",
    }
)
race_averages = (
    all_boats.groupby(race_group_columns)
    .agg(Boats=("Boat", "size"), **{"Field Average Prog": ("Prog", "mean")})
    .reset_index()
)
race_averages["Field Average Prog"] = race_averages["Field Average Prog"].round(2)
race_comparison = (
    race_bests.merge(race_averages, on=race_group_columns)
    .sort_values("Best Prog", ascending=False)
    .reset_index(drop=True)
)

if len(selected_events) > 1:
    matched_summary, matched_detail = build_matched_boat_trends(all_boats, selected_events)
    st.subheader("Matched Boats Across Events")
    st.caption(
        "Matches the exact boat label and exact boat class across every selected event. "
        "Numbered boats such as CAN, CAN1, and CAN2 are treated separately. "
        "The fastest selected race is used when a boat raced more than once at an event. "
        "Negative time change means the boat got faster from the first selected event to the last."
    )

    if matched_summary.empty:
        st.info("No exact boat and boat-class combinations competed in every selected event.")
    else:
        prog_columns = [
            column for column in matched_summary.columns if column.endswith(" Prog")
        ]
        matched_style = (
            matched_summary.style.map(prognostic_text_style, subset=prog_columns)
            .map(trend_text_style, subset=["Time Change (s)"])
            .format(
                {
                    **{column: "{:.2f}" for column in prog_columns},
                    "Time Change (s)": "{:+.2f}",
                    "Prog Change": "{:+.2f}",
                }
            )
        )
        st.dataframe(matched_style, use_container_width=True, hide_index=True)

        with st.expander("Matched boat race details"):
            st.dataframe(
                matched_detail.style.map(
                    prognostic_text_style,
                    subset=["Prog"],
                ).format({"Prog": "{:.2f}", "GPS Seconds": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        matched_csv_col, matched_pdf_col = st.columns(2)
        with matched_csv_col:
            st.download_button(
                "Download Matched Boats CSV",
                data=matched_detail.to_csv(index=False).encode("utf-8"),
                file_name=f"{'_vs_'.join(selected_events)}_matched_boats.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with matched_pdf_col:
            st.download_button(
                "Download Matched Boats PDF",
                data=build_matched_pdf(matched_summary, matched_detail, selected_events),
                file_name=f"{'_vs_'.join(selected_events)}_matched_boats.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

st.subheader("Selected Race Comparison")
st.dataframe(
    race_comparison.style.map(
        prognostic_text_style,
        subset=["Best Prog", "Field Average Prog"],
    ).format({"Best Prog": "{:.2f}", "Field Average Prog": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("All Selected Boats")
st.dataframe(
    all_boats.style.map(prognostic_text_style, subset=["Prog"]).format(
        {"Prog": "{:.2f}", "Rate": "{:.1f}", "Speed": "{:.2f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

summary_columns = [
    "Event",
    "Day / Group",
    "Race",
    "Class",
    "GPS Rank",
    "Boat",
    "Prog",
    "Average Split",
    "Race Time (GPS)",
]
class_bests = (
    all_boats.loc[all_boats.groupby(["Event", "Class"])["Prog"].idxmax(), summary_columns]
    .sort_values("Prog", ascending=False)
    .reset_index(drop=True)
)

with st.expander("Class-best summary by event"):
    st.dataframe(class_bests, use_container_width=True, hide_index=True)

csv_data = all_boats.to_csv(index=False).encode("utf-8")
file_prefix = "_vs_".join(selected_events)
download_col, report_col = st.columns(2)
with download_col:
    st.download_button(
        "Download All Boats CSV",
        data=csv_data,
        file_name=f"{file_prefix}_all_boats.csv",
        mime="text/csv",
        use_container_width=True,
    )
with report_col:
    st.download_button(
        "Download All Boats PDF",
        data=build_pdf(all_boats, selected_events),
        file_name=f"{file_prefix}_all_boats.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
