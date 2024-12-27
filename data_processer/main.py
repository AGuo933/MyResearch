import pandas as pd
from helper_functions import (
    load_scada_data,
    load_scada_logs,
    load_annotations,
    generate_data_overview,
    plot_overview_cockpit,
    plot_summary,
)


def main():
    base_path = "data/"

    # Define file lists for each data type
    scada_files = [
        "Wind-Turbine-SCADA-signals-2016.csv",
        "Wind-Turbine-SCADA-signals-2017_0.csv",
    ]

    log_files = [
        "Wind-Turbines-Logs-2016.csv",
        "Wind Turbines Logs 2017.csv",
    ]

    annotation_files = [
        "Historical-Failure-Logbook-2016.csv",
        "opendata-wind-failures-2017.csv",
    ]

    # Load all data
    dct_scada = load_scada_data(base_path, scada_files)
    dct_logs = load_scada_logs(base_path, log_files)
    dct_annot = load_annotations(base_path, annotation_files)

    # Generate overview
    df_data_overview = generate_data_overview(dct_scada, dct_logs)

    # Plot for each turbine
    for trb_id in list(dct_scada.keys()):
        plot_overview_cockpit(
            base_path, dct_scada, df_data_overview, trb_id, freq_="10T"
        )

        # Plot summary with available annotations
        plot_summary(
            trb_id, df_data_overview.loc[trb_id], dct_annot.get(trb_id, pd.DataFrame())
        )


if __name__ == "__main__":
    main()
