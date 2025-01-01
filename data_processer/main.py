from helper_functions import (
    load_scada_data,
    plot_power_scatter,
    clean_power_curve_data,
    save_cleaned_data,
    analyze_power_correlations,
)


def main():
    base_path = "data/originalData/"

    # Define file lists for each data type
    scada_files = [
        "Wind-Turbine-SCADA-signals-2016.csv",
        "Wind-Turbine-SCADA-signals-2017_0.csv",
    ]

    # Load all data
    dct_scada = load_scada_data(base_path, scada_files)

    # 为每个涡轮机绘制功率散点图
    for trb_id in dct_scada.keys():
        # 清洗数据
        df_clean = clean_power_curve_data(dct_scada[trb_id])

        correlations = analyze_power_correlations(
            df_clean, plot_heatmap=True, trb_id=trb_id
        )

        # 保存清洗后的数据
        save_cleaned_data(df_clean, trb_id)

        # 绘制功率散点图
        plot_power_scatter(
            df_scada=df_clean,
            trb_id=trb_id,
            save_path=f"images/power_curve_turbine_after_clean_{trb_id}.png",
        )


if __name__ == "__main__":
    main()
