import json
import os
from pathlib import Path
import re
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_left_right_y(val):
    """左右打出角の文字列（左=プラス, 右=マイナス）を数値変換"""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if "右" in val_str:
        return -float(val_str.replace("右", ""))
    elif "左" in val_str:
        return float(val_str.replace("左", ""))
    else:
        try:
            return float(val_str)
        except ValueError:
            return 0.0


def load_and_aggregate_trackman_data(root_dir="."):
    """'YYYY-MM-DD range' 形式のフォルダを再帰的に検索して全CSVを統合"""
    date_folder_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s+range")
    root = Path(root_dir)

    matched_dirs = []
    for path in root.rglob("*"):
        if path.is_dir():
            match = date_folder_pattern.search(path.name)
            if match:
                date_str = match.group(1)
                matched_dirs.append((date_str, path))

    matched_dirs.sort(key=lambda x: x[0])

    dfs = []
    for date_str, dir_path in matched_dirs:
        csv_files = sorted(dir_path.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df["date"] = date_str

                if "No." not in df.columns or df["No."].isnull().any():
                    df["session_shot_no"] = range(1, len(df) + 1)
                else:
                    df["session_shot_no"] = df["No."]

                dfs.append(df)
            except Exception as e:
                print(f"Failed to read {csv_file}: {e}")

    if not dfs:
        print("No CSV files found in 'YYYY-MM-DD range' folders.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["global_shot_no"] = range(1, len(full_df) + 1)

    # Club-type の確保
    if "Club-type" not in full_df.columns:
        full_df["Club-type"] = "9i"
    else:
        full_df["Club-type"] = (
            full_df["Club-type"].fillna("9i").astype(str).str.strip()
        )

    if "左右打出角 (度)" in full_df.columns:
        full_df["左右打出角_y"] = full_df["左右打出角 (度)"].apply(
            parse_left_right_y
        )
    else:
        full_df["左右打出角_y"] = 0.0

    if "打出角 (度)" in full_df.columns:
        full_df["打出角_num"] = pd.to_numeric(
            full_df["打出角 (度)"], errors="coerce"
        ).fillna(0.0)
    else:
        full_df["打出角_num"] = 0.0

    if "最高到達点 (yds)" in full_df.columns:
        full_df["最高到達点_num"] = pd.to_numeric(
            full_df["最高到達点 (yds)"], errors="coerce"
        ).fillna(0.0)
    else:
        full_df["最高到達点_num"] = 0.0

    return full_df


def plot_trackman_plotly(
    df,
    height_scale=1.0,
    launch_scale=0.6,
    min_peak_height=3,
    min_launch_angle=5,
    peak_height_samples=[3, 10, 20, 30],
    launch_angle_samples=[5, 10, 20, 30, 40],
):
    """Plotlyによる2段組インタラクティブ散布図"""
    if df.empty:
        print("DataFrame is empty.")
        return None, ""

    def get_club_symbol_and_label(club_name):
        c = club_name.lower()
        if c == "7i":
            return "triangle-up", "7i"
        elif c in ["driver", "1w", "dr"]:
            return "square", "Driver"
        return "circle", club_name

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Carry & Peak Height",
            "Horiz. Launch Angle & Launch Angle",
        ),
    )

    # ツールチップ用テキスト
    hover_top = {
        idx: (
            f"<b>Date:</b> {row['date']}<br>"
            f"<b>Club:</b> {row['Club-type']}<br>"
            f"<b>Shot:</b> #{row['session_shot_no']} (Total #{row['global_shot_no']})<br>"
            f"<b>Carry:</b> {row.get('キャリー (yds)', 'N/A')} yds<br>"
            f"<b>Ball Speed:</b> {row.get('ボールスピード (m/s)', 'N/A')} m/s<br>"
            f"<b>Peak Height:</b> {row['最高到達点_num']} yds"
        )
        for idx, row in df.iterrows()
    }

    hover_bottom = {
        idx: (
            f"<b>Date:</b> {row['date']}<br>"
            f"<b>Club:</b> {row['Club-type']}<br>"
            f"<b>Shot:</b> #{row['session_shot_no']} (Total #{row['global_shot_no']})<br>"
            f"<b>Horiz. Launch:</b> {row['左右打出角_y']:.1f}°<br>"
            f"<b>Ball Speed:</b> {row.get('ボールスピード (m/s)', 'N/A')} m/s<br>"
            f"<b>Launch Angle:</b> {row['打出角_num']:.1f}°"
        )
        for idx, row in df.iterrows()
    }

    # 最小サイズの適用 (clip処理)
    df["peak_height_sizes"] = (
        df["最高到達点_num"].clip(lower=min_peak_height) * height_scale
    )
    df["launch_angle_sizes"] = (
        df["打出角_num"].clip(lower=min_launch_angle) * launch_scale
    )

    # --- 散布図プロット (ax1 & ax2) クラブごとに描画 ---
    unique_clubs = df["Club-type"].unique()
    colorbar_shown = False

    for club in unique_clubs:
        sub = df[df["Club-type"] == club]
        sym, _ = get_club_symbol_and_label(club)

        # 1段目 (Carry)
        show_cbar = not colorbar_shown
        scatter1 = go.Scatter(
            x=sub["global_shot_no"],
            y=sub["キャリー (yds)"],
            mode="markers",
            hoverinfo="text",
            hovertext=[hover_top[idx] for idx in sub.index],
            showlegend=False,
            marker=dict(
                symbol=sym,
                size=sub["peak_height_sizes"],
                color=sub["ボールスピード (m/s)"],
                colorscale="Blues",
                cmin=0,
                cmax=50,
                showscale=show_cbar,
                colorbar=dict(
                    title=dict(
                        text="Ball Speed (m/s)",
                        side="right",
                        font=dict(size=13),
                    ),
                    thickness=15,
                    x=1.02,
                    len=0.85,
                    y=0.5,
                    yanchor="middle",
                )
                if show_cbar
                else None,
                line=dict(width=1, color="navy"),
                opacity=0.8,
            ),
            name="Carry",
        )
        fig.add_trace(scatter1, row=1, col=1)
        colorbar_shown = True

        # 2段目 (左右打出角)
        scatter2 = go.Scatter(
            x=sub["global_shot_no"],
            y=sub["左右打出角_y"],
            mode="markers",
            hoverinfo="text",
            hovertext=[hover_bottom[idx] for idx in sub.index],
            showlegend=False,
            marker=dict(
                symbol=sym,
                size=sub["launch_angle_sizes"],
                color=sub["ボールスピード (m/s)"],
                colorscale="Blues",
                cmin=0,
                cmax=50,
                showscale=False,
                line=dict(width=1, color="navy"),
                opacity=0.8,
            ),
            name="Horiz Launch Angle",
        )
        fig.add_trace(scatter2, row=2, col=1)

    # 最大キャリー値注釈
    if "キャリー (yds)" in df.columns and not df["キャリー (yds)"].empty:
        max_idx = df["キャリー (yds)"].idxmax()
        max_carry = df.loc[max_idx, "キャリー (yds)"]
        max_x = df.loc[max_idx, "global_shot_no"]

        fig.add_annotation(
            x=max_x,
            y=max_carry,
            text=f"<b>Max: {int(round(max_carry))} yds</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="navy",
            ax=20,
            ay=-25,
            font=dict(color="navy", size=12),
            row=1,
            col=1,
        )

    # 中心線 (0度)
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1
    )

    # --- 各セッションの区切り破線 ＆ 90°回転日付背景テキスト ---
    unique_dates = df["date"].unique()
    session_ranges = []
    session_dropdown_buttons = []

    initial_xlim = [0, len(df) + 5]

    session_dropdown_buttons.append(
        dict(
            label="All Dates",
            method="relayout",
            args=[
                {"xaxis.range": initial_xlim, "xaxis2.range": initial_xlim}
            ],
        )
    )

    for i, date_str in enumerate(unique_dates):
        sub = df[df["date"] == date_str]
        start_x = int(sub["global_shot_no"].min())
        end_x = int(sub["global_shot_no"].max())
        s_range = [start_x - 1, end_x + 1]
        session_ranges.append(s_range)

        session_dropdown_buttons.append(
            dict(
                label=f"{date_str} ({len(sub)} shots)",
                method="relayout",
                args=[{"xaxis.range": s_range, "xaxis2.range": s_range}],
            )
        )

        if i > 0:
            sep_x = start_x - 0.5
            fig.add_vline(
                x=sep_x,
                line_dash="dot",
                line_color="rgba(100, 100, 100, 0.6)",
                line_width=1.5,
                row=1,
                col=1,
            )
            fig.add_vline(
                x=sep_x,
                line_dash="dot",
                line_color="rgba(100, 100, 100, 0.6)",
                line_width=1.5,
                row=2,
                col=1,
            )

        text_x = start_x - 0.2

        # 1段目 (Carry) 背景日付テキスト
        fig.add_annotation(
            x=text_x,
            y=0.95,
            xref="x",
            yref="y domain",
            text=f"<b>{date_str}</b>",
            showarrow=False,
            textangle=-90,
            font=dict(size=18, color="rgba(160, 160, 160, 0.45)"),
            xanchor="left",
            yanchor="top",
        )

        # 2段目 (Angle) 背景日付テキスト
        fig.add_annotation(
            x=text_x,
            y=0.95,
            xref="x",
            yref="y2 domain",
            text=f"<b>{date_str}</b>",
            showarrow=False,
            textangle=-90,
            font=dict(size=18, color="rgba(160, 160, 160, 0.45)"),
            xanchor="left",
            yanchor="top",
        )

    # --- フローティング L / R テキスト ---
    fig.add_annotation(
        x=0.01,
        y=0.95,
        xref="x domain",
        yref="y2 domain",
        text="<b>L</b>",
        showarrow=False,
        font=dict(size=22, color="navy"),
        align="left",
    )
    fig.add_annotation(
        x=0.01,
        y=0.05,
        xref="x domain",
        yref="y2 domain",
        text="<b>R</b>",
        showarrow=False,
        font=dict(size=22, color="navy"),
        align="left",
    )

    # --- 凡例1: Club-type 凡例 ---
    ordered_clubs = []
    for std_c in ["9i", "7i", "driver", "1w"]:
        for c in unique_clubs:
            if c.lower() == std_c and c not in ordered_clubs:
                ordered_clubs.append(c)
    for c in unique_clubs:
        if c not in ordered_clubs:
            ordered_clubs.append(c)

    for club in ordered_clubs:
        sym, label = get_club_symbol_and_label(club)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    symbol=sym,
                    size=10,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1.2, color="navy"),
                ),
                name=label,
                legendgroup="club_type",
                legendgrouptitle_text="<b>Club</b>",
                showlegend=True,
            )
        )

    # --- 凡例2: Peak Height 凡例 ---
    for h in peak_height_samples:
        size_val = max(h, min_peak_height) * height_scale
        label_text = (
            f"≤ {min_peak_height} yds" if h <= min_peak_height else f"{h} yds"
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=size_val,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1.2, color="navy"),
                ),
                name=label_text,
                legendgroup="peak_height",
                legendgrouptitle_text="<b>Peak Height</b>",
                showlegend=True,
            )
        )

    # --- 凡例3: Launch Angle 凡例 ---
    for a in launch_angle_samples:
        size_val = max(a, min_launch_angle) * launch_scale
        label_text = (
            f"≤ {min_launch_angle}°" if a <= min_launch_angle else f"{a}°"
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=size_val,
                    color="rgba(0,0,0,0)",
                    line=dict(width=1.2, color="navy"),
                ),
                name=label_text,
                legendgroup="launch_angle",
                legendgrouptitle_text="<b>Launch Angle</b>",
                showlegend=True,
            )
        )

    # --- 軸設定 ---
    fig.update_yaxes(
        title_text="Carry (yds)", range=[0, 160], fixedrange=True, row=1, col=1
    )
    fig.update_yaxes(
        title_text="Horiz. Launch Angle (deg)",
        range=[-35, 35],
        fixedrange=True,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Total Shot # (Horizontal Pan/Zoom Enabled)",
        range=initial_xlim,
        row=2,
        col=1,
    )

    # --- ボタン ＆ ドロップダウン設定 ---
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0.01,
            y=1.13,
            xanchor="left",
            yanchor="top",
            pad=dict(r=4, t=6),
            buttons=[
                dict(label="<", method="skip"),
                dict(label=">", method="skip"),
                dict(label="+", method="skip"),
                dict(label="-", method="skip"),
                dict(
                    label="Pan",
                    method="relayout",
                    args=["dragmode", "pan"],
                ),
                dict(
                    label="Zoom",
                    method="relayout",
                    args=["dragmode", "zoom"],
                ),
                dict(
                    label="Reset",
                    method="relayout",
                    args=[
                        {
                            "xaxis.range": initial_xlim,
                            "xaxis2.range": initial_xlim,
                        }
                    ],
                ),
            ],
        ),
        dict(
            type="dropdown",
            direction="down",
            x=0.38,
            y=1.13,
            xanchor="left",
            yanchor="top",
            pad=dict(r=6, t=6),
            buttons=session_dropdown_buttons,
        ),
    ]

    fig.update_layout(
        title=dict(
            text="TrackMan All Session Analysis",
            font=dict(size=18),
            y=0.98,
        ),
        height=820,
        dragmode="pan",
        updatemenus=updatemenus,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            x=1.08,
            y=0.5,
            yanchor="middle",
            title_font_size=12,
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        margin=dict(t=120, b=60, l=80, r=170),
    )

    # 永続的なグローバルイベントリスナー (何回クリックしても動作)
    post_script = f"""
    (function() {{
        var sessionRanges = {json.dumps(session_ranges)};
        var totalShots = {len(df) + 5};

        document.addEventListener('click', function(e) {{
            var item = e.target.closest('g.updatemenu-item-group, g.updatemenu-button, .updatemenu-button');
            if (!item) return;
            var textEl = item.querySelector('text') || item;
            var txt = (textEl.textContent || '').trim();
            if (['<', '>', '+', '-'].indexOf(txt) === -1) return;

            e.preventDefault();
            e.stopPropagation();

            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis) return;

            var curX = gd._fullLayout.xaxis.range;
            var curCenter = (curX[0] + curX[1]) / 2;
            var curSpan = curX[1] - curX[0];
            var isDefault = (curSpan >= totalShots * 0.7);

            var closestIdx = 0;
            var minDiff = Infinity;
            for (var i = 0; i < sessionRanges.length; i++) {{
                var sCenter = (sessionRanges[i][0] + sessionRanges[i][1]) / 2;
                var diff = Math.abs(curCenter - sCenter);
                if (diff < minDiff) {{
                    minDiff = diff;
                    closestIdx = i;
                }}
            }}

            var target = null;
            if (txt === '>') {{
                if (isDefault) {{
                    target = sessionRanges[0];
                }} else {{
                    var nextIdx = Math.min(sessionRanges.length - 1, closestIdx + 1);
                    target = sessionRanges[nextIdx];
                }}
            }} else if (txt === '<') {{
                if (isDefault) {{
                    target = sessionRanges[sessionRanges.length - 1];
                }} else {{
                    var prevIdx = Math.max(0, closestIdx - 1);
                    target = sessionRanges[prevIdx];
                }}
            }} else if (txt === '+') {{
                var newSpan = Math.max(4, curSpan / 2);
                target = [curCenter - newSpan / 2, curCenter + newSpan / 2];
            }} else if (txt === '-') {{
                var newSpan = curSpan * 2;
                if (newSpan >= totalShots) {{
                    target = [0, totalShots];
                }} else {{
                    target = [Math.max(0, curCenter - newSpan / 2), Math.min(totalShots, curCenter + newSpan / 2)];
                }}
            }}

            if (target) {{
                Plotly.relayout(gd, {{
                    'xaxis.range': target,
                    'xaxis2.range': target
                }});
            }}
        }}, true);
    }})();
    """

    return fig, post_script


import platform
import subprocess

if __name__ == "__main__":
    df_all = load_and_aggregate_trackman_data(".")
    print(
        f"Loaded {len(df_all)} shots from {df_all['date'].nunique() if not df_all.empty else 0} session(s)."
    )

    if not df_all.empty:
        fig, post_js = plot_trackman_plotly(df_all)
        html_file = "golf_range_analysis.html"
        fig.write_html(html_file, post_script=post_js)
        print(f"Saved to {html_file}")

        # Open in default browser without AppleScript errors
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", html_file])
        elif platform.system() == "Windows":
            os.startfile(html_file)
        else:  # Linux
            subprocess.run(["xdg-open", html_file])