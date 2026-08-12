import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_golf_shots(
    df_or_path,
    mode="two_panel",  # 'two_panel' (二段組) または 'single' (上段のみ)
    xlim=(0, 60),
    carry_ylim=(0, 150),
    ball_speed_lim=(0, 50),
    peak_height_samples=[5, 10, 15, 20, 30],
    launch_angle_samples=[10, 20, 30, 40],
    height_scale=10,  # 最高到達点の円サイズ倍率
    launch_scale=5,  # 打出角の円サイズ倍率
    save_path=None,  # 保存先ファイルパス (例: 'output.png')
    title=None,  # 図のタイトル（例: '2026-08-12'）
):
    """ゴルフショットのデータを散布図化する関数

    Parameters:
    -----------
    df_or_path : str or pd.DataFrame
        CSVのファイルパス、または読み込み済みのDataFrame
    mode : str, default 'two_panel'
        'two_panel'（二段組）または 'single'（1段目のみ）
    xlim : tuple, default (0, 60)
        X軸（Shot #）の表示範囲
    carry_ylim : tuple, default (0, 150)
        1段目Y軸（キャリー）の表示範囲
    ball_speed_lim : tuple, default (0, 50)
        カラーバー（ボールスピード）の範囲 [vmin, vmax]
    peak_height_samples : list, default [5, 10, 15, 20, 30]
        1段目の凡例（最高到達点 yds）に表示する数値リスト
    launch_angle_samples : list, default [10, 20, 30, 40]
        2段目の凡例（打出角 °）に表示する数値リスト
    height_scale : float, default 10
        最高到達点を散布図の円面積(s)に変換する倍率
    launch_scale : float, default 5
        打出角を散布図の円面積(s)に変換する倍率
    save_path : str, optional
        画像を保存する場合のファイル名
    title : str, optional
        フィギュア上部に表示するタイトル（例: '2026-08-12'）
    """
    # データの読み込み・コピー
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # No.列の確保 (1から連番)
    if "No." not in df.columns or df["No."].isnull().any():
        df["No."] = range(1, len(df) + 1)

    # 左右打出角の数値変換（左=プラス, 右=マイナス）
    def parse_left_right_y(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip()
        if "右" in val_str:
            return -float(val_str.replace("右", ""))
        elif "左" in val_str:
            return float(val_str.replace("左", ""))
        else:
            return float(val_str)

    if "左右打出角 (度)" in df.columns:
        df["左右打出角_y"] = df["左右打出角 (度)"].apply(parse_left_right_y)
    if "打出角 (度)" in df.columns:
        df["打出角_num"] = pd.to_numeric(df["打出角 (度)"], errors="coerce")

    # スタイル設定
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.family"] = ["Arial"]

    vmin, vmax = ball_speed_lim
    is_two_panel = mode == "two_panel"

    # Figureの作成
    if is_two_panel:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 8), dpi=200, sharex=True
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5), dpi=200)
        ax2 = None

    # --------------------------------------------------
    # 1段目 (ax1): 飛距離・高さ・球速
    # --------------------------------------------------
    scatter1 = ax1.scatter(
        df["No."],
        df["キャリー (yds)"],
        c=df["ボールスピード (m/s)"],
        s=df["最高到達点 (yds)"] * height_scale,
        cmap="Blues",
        alpha=0.8,
        edgecolors="navy",
        linewidths=0.8,
        vmin=vmin,
        vmax=vmax,
    )

    ax1.set_xlabel("Shot #", labelpad=12, fontsize=14)
    ax1.set_ylabel("Carry (yds)", labelpad=14, fontsize=12)
    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(carry_ylim[0], carry_ylim[1])
    ax1.tick_params(labelbottom=True)

    # 1段目の凡例（Peak Height）
    legend_handles1 = [
        plt.scatter(
            [],
            [],
            c="none",
            s=h * height_scale,
            edgecolors="navy",
            linewidths=0.8,
            label=f"{h} yds",
        )
        for h in peak_height_samples
    ]
    ax1.legend(
        handles=legend_handles1,
        title="Peak Height",
        loc="center left",
        bbox_to_anchor=(1.22, 0.5),
        frameon=False,
        labelspacing=0.8,
        borderpad=0.8,
        borderaxespad=0.0,
    )

    # --------------------------------------------------
    # 2段目 (ax2): 左右打出角・打出角 (二段組時のみ)
    # --------------------------------------------------
    if is_two_panel and ax2 is not None:
        scatter2 = ax2.scatter(
            df["No."],
            df["左右打出角_y"],
            c=df["ボールスピード (m/s)"],
            s=df["打出角_num"] * launch_scale,
            cmap="Blues",
            alpha=0.8,
            edgecolors="navy",
            linewidths=0.8,
            vmin=vmin,
            vmax=vmax,
        )

        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_xlabel("Shot #", labelpad=12, fontsize=14)
        ax2.set_ylabel("Horiz. Launch Angle (deg)", labelpad=12, fontsize=12)
        ax2.set_ylim(-35, 35)

        ax2.text(0.02, 0.02, "R", transform=ax2.transAxes, fontsize=18, fontweight="bold", ha="left", va="bottom")
        ax2.text(0.02, 0.98, "L", transform=ax2.transAxes, fontsize=18, fontweight="bold", ha="left", va="top")

        # 2段目の凡例（Launch Angle）
        legend_handles2 = [
            plt.scatter(
                [],
                [],
                c="none",
                s=a * launch_scale,
                edgecolors="navy",
                linewidths=0.8,
                label=f"{a}°",
            )
            for a in launch_angle_samples
        ]
        ax2.legend(
            handles=legend_handles2,
            title="Launch Angle",
            loc="center left",
            bbox_to_anchor=(1.22, 0.5),
            frameon=False,
            labelspacing=0.8,
            borderpad=0.8,
            borderaxespad=0.0,
        )

    # --------------------------------------------------
    # 共通カラーバー（Ball Speed）
    # --------------------------------------------------
    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.80, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(scatter1, cax=cbar_ax)
    cbar.set_label("Ball Speed (m/s)", fontsize=14)

    # 図タイトル（任意）
    if title is not None:
        fig.suptitle(title, y=0.98, fontsize=16)

    # 保存処理
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
    return fig