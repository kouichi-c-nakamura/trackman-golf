import json
import os
from pathlib import Path
import platform
import re
import subprocess
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_left_right_y(val):
    """左右打出角の文字列（左=プラス, 右=マイナス, 欠損・'-'=NaN）を数値変換"""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ["-", "--", "", "nan", "None"]:
        return np.nan
    try:
        if "右" in val_str:
            return -float(val_str.replace("右", "").strip())
        elif "左" in val_str:
            return float(val_str.replace("左", "").strip())
        else:
            return float(val_str)
    except (ValueError, TypeError):
        return np.nan


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

    # 数値列の安全な型変換（'-' 等の欠損値を NaN に強制変換）
    num_cols = ["キャリー (yds)", "ボールスピード (m/s)", "最高到達点 (yds)", "打出角 (度)"]
    for col in num_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

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
        full_df["左右打出角_y"] = np.nan

    if "打出角 (度)" in full_df.columns:
        full_df["打出角_num"] = full_df["打出角 (度)"].fillna(0.0)
    else:
        full_df["打出角_num"] = 0.0

    if "最高到達点 (yds)" in full_df.columns:
        full_df["最高到達点_num"] = full_df["最高到達点 (yds)"].fillna(0.0)
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
    def fmt_val(val, unit=""):
        if pd.isna(val):
            return "N/A"
        if isinstance(val, (int, float, np.floating, np.integer)):
            return f"{val:.1f}{unit}"
        return f"{val}{unit}"

    hover_top = {
        idx: (
            f"<b>Date:</b> {row['date']}<br>"
            f"<b>Club:</b> {row['Club-type']}<br>"
            f"<b>Shot:</b> #{row['session_shot_no']} (Total #{row['global_shot_no']})<br>"
            f"<b>Carry:</b> {fmt_val(row.get('キャリー (yds)'), ' yds')}<br>"
            f"<b>Ball Speed:</b> {fmt_val(row.get('ボールスピード (m/s)'), ' m/s')}<br>"
            f"<b>Peak Height:</b> {fmt_val(row.get('最高到達点 (yds)'), ' yds')}"
        )
        for idx, row in df.iterrows()
    }

    hover_bottom = {
        idx: (
            f"<b>Date:</b> {row['date']}<br>"
            f"<b>Club:</b> {row['Club-type']}<br>"
            f"<b>Shot:</b> #{row['session_shot_no']} (Total #{row['global_shot_no']})<br>"
            f"<b>Horiz. Launch:</b> {fmt_val(row.get('左右打出角_y'), '°')}<br>"
            f"<b>Ball Speed:</b> {fmt_val(row.get('ボールスピード (m/s)'), ' m/s')}<br>"
            f"<b>Launch Angle:</b> {fmt_val(row.get('打出角 (度)'), '°')}"
        )
        for idx, row in df.iterrows()
    }

    # 最小サイズの適用
    df["peak_height_sizes"] = (
        df["最高到達点_num"].clip(lower=min_peak_height) * height_scale
    )
    df["launch_angle_sizes"] = (
        df["打出角_num"].clip(lower=min_launch_angle) * launch_scale
    )

    # --- 散布図プロット (ax1 & ax2) ---
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
    if "キャリー (yds)" in df.columns and not df["キャリー (yds)"].dropna().empty:
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

    # 中心線 (0度) — 赤実線
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="red",
        line_width=1,
        row=2,
        col=1,
        layer="above",
    )

    # --- 各セッションの区切り破線 ＆ 90°回転日付背景テキスト ＆ 統計辞書作成 ---
    unique_dates = df["date"].unique()
    session_ranges = []
    session_dropdown_buttons = []
    session_meta = []
    daily_stats_dict = {}

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

        text_x = start_x - 0.2
        session_meta.append({"date": date_str, "x": text_x})

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

        # --- クラブごとの詳細統計サマリーの生成 ---
        club_stats_blocks = []
        for c_name in sub["Club-type"].unique():
            c_sub = sub[sub["Club-type"] == c_name]
            n_c = len(c_sub)

            # キャリー
            c_carry = c_sub["キャリー (yds)"].dropna()
            c_max_carry = c_carry.max()
            c_avg_carry = c_carry.mean()
            c_sd_carry = c_carry.std()

            carry_str = "N/A"
            if not c_carry.empty:
                sd_part = f" ± {c_sd_carry:.1f}" if pd.notna(c_sd_carry) else ""
                carry_str = f"avg {c_avg_carry:.1f}{sd_part} yds (max {c_max_carry:.1f})"

            # ボールスピード
            c_speed = c_sub["ボールスピード (m/s)"].dropna()
            c_max_speed = c_speed.max()
            c_avg_speed = c_speed.mean()
            c_sd_speed = c_speed.std()

            speed_str = "N/A"
            if not c_speed.empty:
                sd_part = f" ± {c_sd_speed:.1f}" if pd.notna(c_sd_speed) else ""
                speed_str = f"avg {c_avg_speed:.1f}{sd_part} m/s (max {c_max_speed:.1f})"

            # 打出角
            c_launch = c_sub["打出角 (度)"].dropna()
            launch_str = f"{c_launch.mean():.1f}°" if not c_launch.empty else "N/A"

            # 左右打出角
            c_horiz = c_sub["左右打出角_y"].dropna()
            if not c_horiz.empty:
                h_mean = c_horiz.mean()
                if h_mean > 0.3:
                    horiz_str = f"L {abs(h_mean):.1f}°"
                elif h_mean < -0.3:
                    horiz_str = f"R {abs(h_mean):.1f}°"
                else:
                    horiz_str = f"Straight ({h_mean:+.1f}°)"
            else:
                horiz_str = "N/A"

            block = (
                f"<div style='margin-bottom:6px;'>"
                f"<b>【 {c_name} 】</b> ({n_c} shots)<br>"
                f"  • <b>Carry:</b> {carry_str}<br>"
                f"  • <b>Speed:</b> {speed_str}<br>"
                f"  • <b>Launch:</b> avg {launch_str} | <b>Horiz:</b> {horiz_str}"
                f"</div>"
            )
            club_stats_blocks.append(block)

        stats_html = (
            f"<div style='font-family:sans-serif; font-size:13px; line-height:1.4; color:#222;'>"
            f"<div style='font-size:14px; font-weight:bold; border-bottom:1px solid #ccc; padding-bottom:4px; margin-bottom:6px;'>"
            f"📅 Session: {date_str} (Total {len(sub)} shots)</div>"
            + "".join(club_stats_blocks)
            + "</div>"
        )
        daily_stats_dict[date_str] = stats_html

        # 1段目 (Carry) 90° 回転背景日付
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
        # 2段目 (Angle) 90° 回転背景日付
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

    # --- 凡例1: Club-type ---
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

    # --- 凡例2: Peak Height ---
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

    # --- 凡例3: Launch Angle ---
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

    # 永続的なグローバルイベントリスナー ＆ 幾何座標ベースの日付ホバーポップアップ
    post_script = f"""
    (function() {{
        var sessionRanges = {json.dumps(session_ranges)};
        var sessionMeta = {json.dumps(session_meta)};
        var dailyStats = {json.dumps(daily_stats_dict)};
        var totalShots = {len(df) + 5};

        // 1. ポップアップ要素の生成
        var tooltip = document.createElement('div');
        tooltip.id = 'tm-date-tooltip';
        tooltip.style.position = 'fixed';
        tooltip.style.display = 'none';
        tooltip.style.zIndex = '99999';
        tooltip.style.backgroundColor = 'rgba(255, 255, 255, 0.96)';
        tooltip.style.border = '1px solid #78909c';
        tooltip.style.borderRadius = '6px';
        tooltip.style.padding = '10px 14px';
        tooltip.style.boxShadow = '0 4px 14px rgba(0,0,0,0.18)';
        tooltip.style.pointerEvents = 'none';
        tooltip.style.maxWidth = '360px';
        document.body.appendChild(tooltip);

        // 2. グラフ上のマウス移動を座標計算して日付エリアを判定
        document.addEventListener('mousemove', function(e) {{
            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis) return;

            var rect = gd.getBoundingClientRect();
            var fl = gd._fullLayout;
            var mousePixelX = e.clientX - rect.left - fl.xaxis._offset;
            var mousePixelY = e.clientY - rect.top;

            // X軸プロット範囲外
            if (mousePixelX < 0 || mousePixelX > fl.xaxis._length) {{
                tooltip.style.display = 'none';
                return;
            }}

            // 上段プロット or 下段プロットの領域内か判定
            var inRow1 = (mousePixelY >= fl.yaxis._offset && mousePixelY <= fl.yaxis._offset + fl.yaxis._length);
            var inRow2 = (mousePixelY >= fl.yaxis2._offset && mousePixelY <= fl.yaxis2._offset + fl.yaxis2._length);

            if (!inRow1 && !inRow2) {{
                tooltip.style.display = 'none';
                return;
            }}

            // ピクセルX座標をShot番号(Data X)に変換
            var xRange = fl.xaxis.range;
            var dataX = xRange[0] + (mousePixelX / fl.xaxis._length) * (xRange[1] - xRange[0]);

            // 画面上で文字幅(約20px)に相当するData X許容幅を算出
            var thresholdDataX = Math.max(0.5, (20 / fl.xaxis._length) * (xRange[1] - xRange[0]));

            var matchedDate = null;
            for (var i = 0; i < sessionMeta.length; i++) {{
                if (Math.abs(dataX - sessionMeta[i].x) <= thresholdDataX) {{
                    matchedDate = sessionMeta[i].date;
                    break;
                }}
            }}

            if (matchedDate && dailyStats[matchedDate]) {{
                tooltip.innerHTML = dailyStats[matchedDate];
                tooltip.style.display = 'block';
                var posX = e.clientX + 15;
                var posY = e.clientY + 15;
                if (posX + 360 > window.innerWidth) posX = e.clientX - 375;
                if (posY + 260 > window.innerHeight) posY = e.clientY - 270;
                tooltip.style.left = posX + 'px';
                tooltip.style.top = posY + 'px';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }});

        // 3. ナビゲーションボタン操作
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


if __name__ == "__main__":
    df_all = load_and_aggregate_trackman_data(".")
    print(
        f"Loaded {len(df_all)} shots from {df_all['date'].nunique() if not df_all.empty else 0} session(s)."
    )

    if not df_all.empty:
        fig, post_js = plot_trackman_plotly(df_all)
        html_file = "index.html"
        fig.write_html(html_file, post_script=post_js)
        print(f"Saved to {html_file}")

        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", html_file])
        elif platform.system() == "Windows":
            os.startfile(html_file)
        else:  # Linux
            subprocess.run(["xdg-open", html_file])