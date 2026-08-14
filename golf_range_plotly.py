import os
from pathlib import Path
import re
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
    """'YYYY-MM-DD range' 形式のフォルダを再帰的に検索して全CSVを統合し、

    date列と全体ショット番号(global_shot_no)を追加する
    """
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
    min_peak_height=3,  # 最高到達点の下限値 (yds)
    min_launch_angle=5,  # 打出角の下限値 (deg)
    peak_height_samples=[10, 20, 30],
    launch_angle_samples=[10, 20, 30, 40],
):
    """Plotlyによる2段組インタラクティブ散布図 (最小バブルサイズ下限設定付き)"""
    if df.empty:
        print("DataFrame is empty.")
        return None

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
    hover_top = [
        f"<b>Date:</b> {row['date']}<br>"
        f"<b>Shot:</b> #{row['session_shot_no']} (Total #{row['global_shot_no']})<br>"
        f"<b>Carry:</b> {row.get('キャリー (yds)', 'N/A')} yds<br>"
        f"<b>Ball Speed:</b> {row.get('ボールスピード (m/s)', 'N/A')} m/s<br>"
        f"<b>Peak Height:</b> {row['最高到達点_num']} yds"
        for _, row in df.iterrows()
    ]

    hover_bottom = [
        f"<b>Date:</b> {row['date']}<br>"
        f"<b>Shot:</b> #{row['session_shot_no']} (Total #{row['global_shot_no']})<br>"
        f"<b>Horiz. Launch:</b> {row['左右打出角_y']:.1f}°<br>"
        f"<b>Ball Speed:</b> {row.get('ボールスピード (m/s)', 'N/A')} m/s<br>"
        f"<b>Launch Angle:</b> {row['打出角_num']:.1f}°"
        for _, row in df.iterrows()
    ]

    # --- 最小サイズの適用 (clip処理) ---
    peak_height_sizes = (
        df["最高到達点_num"].clip(lower=min_peak_height) * height_scale
    )
    launch_angle_sizes = (
        df["打出角_num"].clip(lower=min_launch_angle) * launch_scale
    )

    # --- 1段目 (Carry) ---
    scatter1 = go.Scatter(
        x=df["global_shot_no"],
        y=df["キャリー (yds)"],
        mode="markers",
        hoverinfo="text",
        hovertext=hover_top,
        showlegend=False,
        marker=dict(
            size=peak_height_sizes,
            color=df["ボールスピード (m/s)"],
            colorscale="Blues",
            cmin=0,
            cmax=50,
            showscale=True,
            colorbar=dict(
                title="Ball Speed (m/s)", x=1.02, len=0.85, y=0.5, yanchor="middle"
            ),
            line=dict(width=1, color="navy"),
            opacity=0.8,
        ),
        name="Carry",
    )
    fig.add_trace(scatter1, row=1, col=1)

    # 最大キャリー値注釈
    if "キャリー (yds)" in df.columns and not df["キャリー (yds)"].empty:
        max_idx = df["キャリー (yds)"].idxmax()
        max_carry = df.loc[max_idx, "キャリー (yds)"]
        max_x = df.loc[max_idx, "global_shot_no"]
        max_date = df.loc[max_idx, "date"]

        fig.add_annotation(
            x=max_x,
            y=max_carry,
            text=f"<b>Max: {int(round(max_carry))} yds ({max_date})</b>",
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

    # --- 2段目 (左右打出角) ---
    scatter2 = go.Scatter(
        x=df["global_shot_no"],
        y=df["左右打出角_y"],
        mode="markers",
        hoverinfo="text",
        hovertext=hover_bottom,
        showlegend=False,
        marker=dict(
            size=launch_angle_sizes,
            color=df["ボールスピード (m/s)"],
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

    # 中心線 (0度)
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1
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

    # --- 日付区切り線 ＆ 日付テキスト ---
    unique_dates = df["date"].unique()
    for i, date_str in enumerate(unique_dates):
        sub = df[df["date"] == date_str]
        start_x = sub["global_shot_no"].min()
        end_x = sub["global_shot_no"].max()
        center_x = (start_x + end_x) / 2

        if i > 0:
            sep_x = start_x - 0.5
            fig.add_vline(
                x=sep_x,
                line_dash="dot",
                line_color="rgba(120, 120, 120, 0.5)",
                line_width=1.5,
            )

        fig.add_annotation(
            x=center_x,
            y=1.03,
            xref="x",
            yref="paper",
            text=f"<b>{date_str}</b>",
            showarrow=False,
            font=dict(size=12, color="#222222"),
            align="center",
        )

    # --- 円サイズ凡例 ---
    for h in peak_height_samples:
        size_val = max(h, min_peak_height) * height_scale
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
                name=f"{h} yds",
                legendgroup="peak_height",
                legendgrouptitle_text="<b>Peak Height</b>",
                showlegend=True,
            )
        )

    for a in launch_angle_samples:
        size_val = max(a, min_launch_angle) * launch_scale
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
                name=f"{a}°",
                legendgroup="launch_angle",
                legendgrouptitle_text="<b>Launch Angle</b>",
                showlegend=True,
            )
        )

    # --- 軸設定（Y軸は固定） ---
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
        range=[0, len(df) + 5],
        row=2,
        col=1,
    )

    # --- ボタン設定 ---
    initial_xlim = [0, len(df) + 5]
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0.01,
            y=1.12,
            xanchor="left",
            yanchor="top",
            pad=dict(r=10, t=10),
            buttons=[
                dict(
                    label="Mode: Pan (Horizontal)",
                    method="relayout",
                    args=["dragmode", "pan"],
                ),
                dict(
                    label="Mode: Zoom (Horizontal)",
                    method="relayout",
                    args=["dragmode", "zoom"],
                ),
                dict(
                    label="Reset View",
                    method="relayout",
                    args=[
                        {
                            "xaxis.range": initial_xlim,
                            "xaxis2.range": initial_xlim,
                        }
                    ],
                ),
            ],
        )
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
            x=1.14,
            y=0.5,
            yanchor="middle",
            title_font_size=12,
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        margin=dict(t=120, b=60, l=80, r=220),
    )

    return fig


if __name__ == "__main__":
    df_all = load_and_aggregate_trackman_data(".")
    print(
        f"Loaded {len(df_all)} shots from {df_all['date'].nunique() if not df_all.empty else 0} session(s)."
    )

    if not df_all.empty:
        fig = plot_trackman_plotly(df_all)
        fig.show()