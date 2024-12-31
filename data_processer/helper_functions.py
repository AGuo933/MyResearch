import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import windrose
from windrose import WindroseAxes
from tabulate import tabulate

from IPython.display import display, HTML
from matplotlib.cm import ScalarMappable
from sklearn.neighbors import LocalOutlierFactor


def load_scada_data(base_path: str, lst_files: list) -> dict:
    """Loads and processes SCADA data from multiple CSV files.

    Args:
        base_path: Directory path containing CSV files (with trailing /)
        lst_files: List of CSV files to load
    Returns:
        Dict mapping turbine IDs to their DataFrames with UTC timestamps
    """

    print(f"Loading SCADA signals...")

    dct_scada = {}

    if check_xlsx(lst_files):
        return

    df_scada_all = pd.DataFrame()

    # load files into df
    for i, file in enumerate(lst_files, 1):
        df_temp = pd.read_csv(base_path + file, index_col="Timestamp")
        df_temp.index = pd.to_datetime(df_temp.index, utc=True)
        df_scada_all = pd.concat([df_scada_all, df_temp])

    # organize data by turbine
    for trb_id in np.unique(df_scada_all.Turbine_ID):
        df_turbine = df_scada_all[df_scada_all["Turbine_ID"] == trb_id]
        df_turbine = df_turbine.drop("Turbine_ID", axis="columns")
        df_turbine.index.name = trb_id
        df_turbine["power"] = df_turbine.Grd_Prod_Pwr_Avg
        df_turbine["wind_speed"] = df_turbine.Amb_WindSpeed_Avg
        df_turbine["wind_direction"] = df_turbine.Amb_WindDir_Abs_Avg

        dct_scada[trb_id] = df_turbine[
            ~df_turbine.index.duplicated(keep="first")
        ].sort_index()

    return dct_scada


def load_scada_logs(base_path: str, lst_files: list) -> dict:
    """
    Load, unify, and process SCADA log files.

    Parameters:
        base_path (str): Base directory path for the log files.
        lst_files (list): List of log files to load.

    Returns:
        dict: A dictionary where keys are turbine IDs and values are DataFrames of logs for each turbine.
    """

    print(f"Loading SCADA logs...")

    dct_logs = {}

    if check_xlsx(lst_files):
        return

    dct_unify_colnames = dict(
        {
            "col_time": ["TimeDetected", "Time_Detected"],
            "col_trb_id": ["UnitTitle", "Turbine_Identifier"],
            "col_message": ["Remark", "Remark"],
        }
    )

    df_logs_all = pd.DataFrame()
    for i, path_ in enumerate(lst_files):
        df_ = pd.read_csv(
            base_path + path_, index_col=dct_unify_colnames["col_time"][i]
        )
        df_.index = pd.to_datetime(df_.index, utc=True)
        columns = [dct_unify_colnames[col_name][i] for col_name in dct_unify_colnames]
        col_names = [col_name[4:] for col_name in dct_unify_colnames]
        df = pd.DataFrame(
            np.array(df_[columns[1:]]), columns=col_names[1:], index=df_.index
        ).sort_index()
        df_logs_all = pd.concat([df_logs_all, df])

    df_logs_all = df_logs_all.loc[
        pd.notnull(df_logs_all.index)
    ].dropna()  # drop all timestamps with missing values

    # sort by turbine
    dct_logs = {trb_id: group for trb_id, group in df_logs_all.groupby("trb_id")}

    return dct_logs


def load_annotations(base_path: str, lst_files: list) -> dict:
    """Loads and processes SCADA data from multiple CSV files.

    Args:
        base_path: Directory path containing CSV files (with trailing /)
        lst_files: List of annotation files to load
    Returns:
        Dict mapping turbine IDs to their DataFrames with UTC timestamps
    """

    print(f"Loading annotations...")

    dct_fail = dict()

    if check_xlsx(lst_files):
        return

    df_fail_all = pd.DataFrame()
    # load files into df
    for i, path_ in enumerate(lst_files):
        df_ = pd.read_csv(base_path + path_, index_col="Timestamp", parse_dates=True)
        df_fail_all = pd.concat([df_fail_all, df_])

    df_fail_all = df_fail_all.dropna().sort_index()

    # sort by turbine
    dct_fail = {trb_id: group for trb_id, group in df_fail_all.groupby("Turbine_ID")}

    return dct_fail


def load_community_annotations(path_: str) -> dict:
    """
    Load and process community annotations from a CSV file.

    Args:
        path_ (str): Path to the community annotation CSV file.

    Returns:
        dict: A dictionary mapping turbine IDs to their respective DataFrames with UTC timestamps.
    """
    print("Loading community annotations...")

    # Read the CSV file into a DataFrame
    df_com_all = (
        pd.read_csv(path_, index_col="annot_id", parse_dates=True).dropna().sort_index()
    )

    # Group by turbine_id and store in a dictionary
    dct_community = {
        trb_id: group for trb_id, group in df_com_all.groupby("turbine_id")
    }

    # Extract dataset name from the file path and attach to dataframes
    dct_community["dataset"] = path_.split("/")[-1].split("_")[0]

    for key_ in dct_community.keys():
        if key_ == "dataset":
            continue
        df_ = dct_community[key_]
        df_.dataset = dct_community["dataset"]
        dct_community[key_] = df_

    return dct_community


def generate_data_overview(dct_scada: dict, dct_logs: dict = {}) -> pd.DataFrame:
    """
    Generate an overview of turbine data metrics from SCADA data.

    Args:
        dct_scada (dict): Dictionary mapping turbine IDs to their SCADA data (pandas DataFrames).
        ar_trb_id (list): List of turbine IDs to process.

    Returns:
        pd.DataFrame: DataFrame summarizing metrics for each turbine, including:
                      - Number of variables and datapoints.
                      - Data timestamps (first and last).
                      - Maximum possible datapoints, missing datapoints, and uptime.
                      - Power produced (MWh).
    """
    print("SCADA DATA OVERVIEW")
    ar_trb_id = list(dct_scada.keys())
    df_data_overview = pd.DataFrame()

    for trb_id in ar_trb_id:
        df_scada_trb = dct_scada[trb_id]
        no_vars = len(df_scada_trb.columns)
        no_datapoints = len(df_scada_trb.index)
        time_first, time_last = (
            np.sort(df_scada_trb.index)[0],
            np.sort(df_scada_trb.index)[-1],
        )

        max_datapoints = (time_last - time_first) / np.timedelta64(10, "m")
        datapoints_missing = int(max_datapoints - no_datapoints)
        data_uptime = np.round(no_datapoints / max_datapoints, 3)

        time_first_str = pd.to_datetime(time_first).strftime("%Y-%m-%d %H:%M")
        time_last_str = pd.to_datetime(time_last).strftime("%Y-%m-%d %H:%M")

        power_produced = df_scada_trb.power.sum() / 6000

        # Store the metrics in the DataFrame
        df_data_overview.loc[trb_id, "# Variables"] = no_vars
        df_data_overview.loc[trb_id, "First Timestamp"] = time_first_str
        df_data_overview.loc[trb_id, "Last Timestamp"] = time_last_str
        df_data_overview.loc[trb_id, "# Datapoints"] = no_datapoints
        df_data_overview.loc[trb_id, "Datapoints missing"] = int(datapoints_missing)
        df_data_overview.loc[trb_id, "Data uptime"] = f"{data_uptime*100} %"
        df_data_overview.loc[trb_id, "Energy Produced Total (MWh)"] = np.round(
            power_produced
        )
        df_data_overview.loc[trb_id, "Capacity Factor)"] = np.round(
            power_produced
            / (8760 * ((time_last - time_first) / np.timedelta64(1, "Y")) * 2),
            2,
        )
        df_data_overview.loc[trb_id, "FLH (h)"] = np.round(
            power_produced / ((time_last - time_first) / np.timedelta64(1, "Y") * 2)
        )

        if dct_logs != {}:
            df_log_trb = dct_logs[trb_id]

            df_data_overview.loc[trb_id, "# Log entries"] = len(df_log_trb)

    max_cols = 4
    if len(df_data_overview.index) < max_cols:
        print(tabulate(df_data_overview.T, headers="keys", tablefmt="pretty"))
    else:
        for i in range((len(df_data_overview.index) // max_cols) + 1):
            print(
                tabulate(
                    df_data_overview.iloc[i * max_cols : (i + 1) * max_cols, :].T,
                    headers="keys",
                    tablefmt="pretty",
                )
            )

    return df_data_overview


def get_last_n_logs(df_logs, timestamp, n=5):
    t_start = timestamp - pd.Timedelta(1, "d")
    t_stop = timestamp

    last_n_logs = df_logs.loc[t_start:t_stop].iloc[-n:, :]

    return last_n_logs


def check_xlsx(file_list):
    xlsx_files = [file for file in file_list if file.endswith(".xlsx")]
    is_xlsx = False
    # If there are any .xlsx files, raise a warning
    if xlsx_files:
        is_xlsx = True
        print(f"The following .xlsx files were found:")
        for file_ in xlsx_files:
            print(file_)

        print(
            f"--> please, save all .xlsx-files as .csv-files to ensure faster data import!"
        )
    return is_xlsx


def plot_summary(trb_id, df_data_overview, dct_ann_owner={}):
    """
    Function to neatly display KPI summary and annotations for a given trb_id.

    Parameters:
    trb_id (str or int): Identifier for the data of interest.
    df_data_overview (pd.DataFrame): DataFrame containing data overview information.
    dct_fail_trb (dict): Dictionary containing owner annotations for each trb_id.
    """

    def print_section(title, content, is_tabular=False):
        print("=" * 110)
        print(f"\033[1m {title} \033[0m")
        print("=" * 110)

        if len(content) == 0:
            print("No annotations...")
        elif is_tabular:
            print(tabulate(content, headers="keys"))
        else:
            print(content)
        print("\n" * 2)

    print_section(
        f"KPI Summary {trb_id}", pd.DataFrame(df_data_overview), is_tabular=True
    )
    print_section(f"Owner Annotations {trb_id}", dct_ann_owner, is_tabular=True)

    print("\n" * 3)


# Function to prepare the power curve
def prepare_power_curve(path_):
    df_powercurve = pd.read_csv(path_, index_col=0)

    df_pc = pd.DataFrame(
        index=[np.round(i, 2) for i in np.arange(0, 30, 0.01)], columns=["power_norm"]
    )
    df_pc[df_pc.index < df_powercurve.index[0]] = 0
    df_pc[df_pc.index > df_powercurve.index[-1]] = 0
    df_pc.loc[df_powercurve.index] = df_powercurve.values.reshape(-1, 1)
    return df_pc.astype(float).interpolate(method="linear")


# Function to plot the wind rose
def plot_wind_rose(fig, wd, ws):
    ax_windrose = WindroseAxes.from_ax(fig=fig, rect=[0.55, 0.55, 0.4, 0.4])
    ax_windrose.bar(wd, ws, normed=True, opening=0.8)
    ax_windrose.set_title("Wind Rose")
    ax_windrose.set_legend(loc="upper left")


# Function to plot the power curve and actual data


def plot_power_curve(ax, wind_speeds, actual_powers, expected_powers, df_pc):
    distances = actual_powers - expected_powers
    norm = plt.Normalize(distances.min(), distances.max())
    colors = plt.cm.viridis(norm(distances))

    scatter = ax.scatter(
        wind_speeds,
        actual_powers,
        c=colors,
        alpha=0.25,
        edgecolor="black",
        label="Measured",
    )
    ax.plot(
        df_pc.index, df_pc["power_norm"], c="k", label="Standard PC"
    )  # Plot the power curve
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cbar.set_label("Distance from Power Curve")

    ax.set_xlabel("Nacelle measured Wind Speed (m/s)")
    ax.set_ylabel("Power Output (kW)")
    ax.set_title("Measured Power Curve")
    ax.grid(True)
    ax.legend()


# Function to plot data availability
def plot_data_availability(ax, time_index, full_time_index, available, data_uptime):
    ax.plot(full_time_index, available, c="grey")

    # Detect changes in availability
    change_indices = np.where(np.diff(available.flatten()) != 0)[0]

    # Iterate through the change indices to mark gaps in data availability
    for idx in range(len(change_indices)):
        if (
            available[change_indices[idx]] == 1
            and available[change_indices[idx] + 1] == 0
        ):
            ax.axvline(
                full_time_index[change_indices[idx]],
                color="red",
                linestyle="--",
                linewidth=1,
            )
            end_idx = (
                change_indices[idx + 1]
                if idx + 1 < len(change_indices)
                else len(full_time_index) - 1
            )
            ax.fill_between(
                full_time_index[change_indices[idx] : end_idx + 1],
                0,
                1,
                color="lightcoral",
                alpha=0.5,
                label="Data Unavailable" if idx == 0 else "",
            )
        elif (
            available[change_indices[idx]] == 0
            and available[change_indices[idx] + 1] == 1
        ):
            ax.axvline(
                full_time_index[change_indices[idx] + 1],
                color="red",
                linestyle="--",
                linewidth=1,
            )

    ax.set_xlim(full_time_index[0], full_time_index[-1])
    ax.set_ylim(0, 1.025)
    # ax.set_aspect(1)
    ax.fill_between(
        full_time_index,
        0,
        1,
        where=available.flatten() == 1,
        color="forestgreen",
        alpha=0.2,
        label="Data Available",
    )
    ax.set_title(f"Data Availability {data_uptime}")
    ax.set_ylabel("Availability")
    ax.legend()


def plot_overview_cockpit(base_path, dct_scada, df_data_overview, trb_id, freq_="10T"):
    df_pc = prepare_power_curve(base_path + "powercurve.csv")
    df_pc = df_pc / 2000 * dct_scada[trb_id]["power"].max()

    # Create images directory if it doesn't exist
    images_path = os.path.join(os.path.dirname(base_path.rstrip("/")), "images")
    os.makedirs(images_path, exist_ok=True)

    # Create a new figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))

    # Left-align the suptitle
    plt.suptitle(f"Overview-Cockpit Turbine {trb_id}", fontsize=16, fontweight="bold")

    axes[0, 1].remove()  # Remove the top-right subplot
    axes[1, 1].remove()  # Remove the bottom-right subplot
    axes[1, 0].remove()

    # Wind Rose Plot
    wd = dct_scada[trb_id]["wind_direction"].values
    ws = dct_scada[trb_id]["wind_speed"].values
    plot_wind_rose(fig, wd, ws)

    # Power Curve Plot
    df_scada = dct_scada[trb_id]
    wind_speeds = df_scada["wind_speed"].values
    actual_powers = df_scada.power.values
    expected_powers = np.array(
        [df_pc.loc[np.round(speed, 2)].values[0] for speed in wind_speeds]
    )

    plot_power_curve(axes[0, 0], wind_speeds, actual_powers, expected_powers, df_pc)

    # Data Availability Plot
    time_index = dct_scada[trb_id].index
    time_first, time_last = np.min(time_index), np.max(time_index)
    full_time_index = pd.date_range(time_first, time_last, freq=freq_)
    available = np.array([int(time in time_index) for time in full_time_index])

    # Add new subplot for data availability
    ax_data_avail = fig.add_subplot(212)
    plot_data_availability(
        ax_data_avail,
        time_index,
        full_time_index,
        available,
        df_data_overview.loc[trb_id, "Data uptime"],
    )

    # Adjust layout and save figure
    plt.tight_layout()
    fig_path = os.path.join(images_path, f"overview_cockpit_turbine_{trb_id}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print(f"Saved figure for turbine {trb_id} to {fig_path}")


def plot_power_scatter(df_scada: pd.DataFrame, trb_id: str, save_path: str = None):
    """
    绘制风力涡轮机的功率散点图。

    Args:
        df_scada (pd.DataFrame): 包含风速和功率数据的DataFrame
        trb_id (str): 涡轮机ID
        save_path (str, optional): 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    wind_speeds = df_scada["wind_speed"].values
    actual_powers = df_scada.power.values

    # 读取并处理标准功率曲线
    df_pc = prepare_power_curve("data/powercurve.csv")
    # 根据实际涡轮机的最大功率调整标准功率曲线
    df_pc_scaled = df_pc / 2000 * df_scada["power"].max()

    # 计算期望功率和偏差
    expected_powers = np.array(
        [df_pc_scaled.loc[np.round(speed, 2)].values[0] for speed in wind_speeds]
    )
    distances = actual_powers - expected_powers

    # 绘制散点图
    norm = plt.Normalize(distances.min(), distances.max())
    scatter = ax.scatter(
        wind_speeds,
        actual_powers,
        c=distances,
        cmap="viridis",
        alpha=0.25,
        edgecolor="black",
        label="Measured",
    )

    # 添加颜色条和标准功率曲线
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Distance from Power Curve")
    ax.plot(df_pc_scaled.index, df_pc_scaled["power_norm"], c="k", label="Standard PC")

    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power Output (kW)")
    ax.set_title(f"Power Curve - Turbine {trb_id}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def clean_power_curve_data(
    df_scada: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.05
):
    """
    清洗风机运行数据，去除异常值。

    Args:
        df_scada (pd.DataFrame): 包含风速和功率数据的DataFrame
        n_neighbors (int, optional): KNN算法的邻居数量，默认为5
        contamination (float, optional): 预期的异常值比例，默认为0.1

    Returns:
        pd.DataFrame: 清洗后的数据
    """
    # 1. 移除包含NaN的行
    df_clean = df_scada.dropna().copy()

    # 2. 移除功率小于等于0的数据
    df_clean = df_clean[df_clean["power"] > 0]

    # 3. 准备用于异常检测的特征
    X = df_clean[["wind_speed", "power"]].values

    # 4. 使用LOF进行异常检测
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    # 预测结果：1表示正常值，-1表示异常值
    y_pred = lof.fit_predict(X)

    # 5. 保留正常值
    df_clean = df_clean[y_pred == 1]

    # 6. 按时间排序
    df_clean = df_clean.sort_index()

    return df_clean


def save_cleaned_data(
    df_clean: pd.DataFrame, trb_id: str, save_path: str = "data/cleaned_data"
):
    """
    保存清洗后的数据到CSV文件，只保留指定的变量。

    Args:
        df_clean (pd.DataFrame): 已经清洗好的DataFrame数据
        trb_id (str): 涡轮机ID
        save_path (str, optional): 保存路径，默认为'data/cleaned_data'
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)

    # 选择要保存的列
    selected_columns = [
        # 风速相关
        "Amb_WindSpeed_Avg",
        "Amb_WindSpeed_Max",
        "Amb_WindSpeed_Min",
        "Amb_WindSpeed_Est_Avg",
        "wind_speed",
        # 风向相关
        "Amb_WindDir_Abs_Avg",
        "Amb_WindDir_Relative_Avg",
        "wind_direction",
        # 机舱方向
        "Nac_Direction_Avg",
        # 桨距角
        "Blds_PitchAngle_Min",
        "Blds_PitchAngle_Max",
        "Blds_PitchAngle_Avg",
        "Blds_PitchAngle_Std",
        # 发电机轴承温度
        "Gen_Bear_Temp_Avg",
        "Gen_Bear2_Temp_Avg",
        # 舱内温度
        "Nac_Temp_Avg",
        # 功率放在最后
        "power",
    ]

    # 检查哪些列实际存在于数据中
    available_columns = [col for col in selected_columns if col in df_clean.columns]

    # 创建新的DataFrame，保持时间索引
    df_save = df_clean[available_columns].copy()

    # 构建文件名
    file_name = f"turbine_{trb_id}_cleaned.csv"
    file_path = os.path.join(save_path, file_name)

    # 保存为CSV，包含时间索引
    df_save.to_csv(file_path)

    print(f"Saved cleaned data for turbine {trb_id} to {file_path}")
    print(f"Saved columns: {available_columns}")


def analyze_power_correlations(
    df_clean: pd.DataFrame,
    power_col: str = "power",
    plot_heatmap: bool = True,
    trb_id: str = None,
) -> pd.DataFrame:
    """
    计算所有变量与功率的斯皮尔曼相关系数并可选择性地绘制热力图。

    Args:
        df_clean (pd.DataFrame): 清洗后的数据
        power_col (str): 功率列的名称，默认为'power'
        plot_heatmap (bool): 是否绘制热力图，默认为True
        trb_id (str): 涡轮机ID，用于保存文件名，默认为None

    Returns:
        pd.DataFrame: 包含相关系数的DataFrame，按相关性绝对值降序排列
    """
    # 计算所有列与功率的斯皮尔曼相关系数
    correlations = pd.DataFrame()
    correlations["variable"] = df_clean.columns
    correlations["correlation"] = [
        df_clean[col].corr(df_clean[power_col], method="spearman")
        for col in df_clean.columns
    ]

    # 移除功率列本身的相关系数
    correlations = correlations[correlations["variable"] != power_col]

    # 按相关系数绝对值降序排列
    correlations["abs_correlation"] = correlations["correlation"].abs()
    correlations = correlations.sort_values("abs_correlation", ascending=False)
    correlations = correlations.drop("abs_correlation", axis=1)

    # 打印结果
    print("\n相关性分析结果:")
    print("================")
    print(tabulate(correlations, headers="keys", tablefmt="pretty", floatfmt=".3f"))

    # 绘制热力图
    if plot_heatmap:
        plt.figure(figsize=(12, 8))

        # 创建相关性矩阵
        heatmap_data = pd.DataFrame(
            index=correlations["variable"],
            columns=["Power Correlation"],
            data=correlations["correlation"].values,
        )

        # 绘制热力图，移除数值标注
        sns.heatmap(
            heatmap_data,
            annot=False,  # 设置为False以移除数值标注
            cmap="coolwarm",
            center=0,
            square=True,
            vmin=-1,
            vmax=1,
        )
        plt.title(f"Power Correlation Heatmap - Turbine {trb_id}")
        plt.tight_layout()

        # 创建images文件夹（如果不存在）
        os.makedirs("images", exist_ok=True)

        # 保存图片
        filename = f"power_correlation_heatmap_{trb_id}.png"
        plt.savefig(f"images/{filename}", dpi=300, bbox_inches="tight")
        plt.close()

    return correlations
