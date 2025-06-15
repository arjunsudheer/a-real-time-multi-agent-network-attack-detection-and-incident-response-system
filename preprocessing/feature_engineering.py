from csv import Error
import pandas as pd
import numpy as np
from typing import Callable


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    engineer_features engineers new features from the network flow DataFrame using rolling time windows.

    Converts the protocol, Src Port, and Dst Port features to categorical representations. Drops the
    Timestamp, Src Port, Dst Port, and Protocol features after calculating the engineered features.

    Args:
        df (pd.DataFrame): The input DataFrame, to sort and add the engineered features
        to.

    Returns:
        pd.DataFrame: The DataFrame with added engineered features.
    """

    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        prepare_dataframe prepares the DataFrame by converting the 'Timestamp' column to datetime
        and sorting the DataFrame by 'Timestamp'.

        Args:
        df (pd.DataFrame): The input DataFrame to sort by 'Timestamp' and have
        'Timestamp' as datetime.

        Returns:
            pd.DataFrame: The prepared DataFrame sorted by 'Timestamp'.
        """
        print("Preparing DataFrame: Converting Timestamp and Sorting.")
        # Convert 'Timestamp' to datetime objects
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True)

        # Sort the DataFrame by 'Timestamp' for time-series operations
        df = df.sort_values(by="Timestamp").reset_index(drop=True)
        df["__idx__"] = df.index
        print("DataFrame preparation complete.")
        return df

    def rolling_feature(
        df_group: pd.DataFrame,
        col: str,
        window: str,
        agg_func: Callable,
        feature_name: str,
    ) -> pd.DataFrame:
        """
        rolling_feature computes a rolling aggregation on a specified column within a time-based window
        for a group of data, preserving the row index for later reassignment.

        Args:
            df_group (pd.DataFrame): A subset of the original DataFrame (usually grouped by an ID).
            col (str): The column name on which to perform the aggregation.
            window (str or pd.Timedelta): The size of the time-based rolling window (e.g., "1s", "5min").
            agg_func (Callable): A function like np.mean, np.sum, etc. to apply to the windowed data.
            feature_name (str): The name of the new feature column to generate.

        Returns:
            pd.DataFrame: A DataFrame with two columns:
                      - "__idx__": Original row index from df_group
                      - feature_name: The aggregated value for each timestamp
        """
        results = []
        timestamps = df_group["Timestamp"]
        for i in range(len(df_group)):
            window_end = timestamps.iloc[i]
            window_start = window_end - pd.to_timedelta(window)
            window_data = df_group[
                (timestamps > window_start) & (timestamps <= window_end)
            ][col]
            val = agg_func(window_data) if not window_data.empty else np.nan
            results.append((df_group["__idx__"].iloc[i], val))
        return pd.DataFrame(results, columns=["__idx__", feature_name])

    def apply_rolling(
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        new_col: str,
        agg_func: Callable,
        window: str = "1s",
    ) -> None:
        """
        apply_rolling applies a time-based rolling aggregation to a DataFrame grouped by a key column.
        The result is added as a new column to the original DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame with time-series data.
            group_col (str): Column name to group by (e.g., device ID, user ID).
            value_col (str): The column on which to compute the rolling aggregation.
            new_col (str): The name of the new column to store the aggregated result.
            agg_func (Callable): The aggregation function to apply.
            window (str or pd.Timedelta): The time window size for the rolling computation.
        """
        result = (
            df.sort_values("Timestamp")
            .groupby(group_col, group_keys=False)
            .apply(lambda g: rolling_feature(g, value_col, window, agg_func, new_col))
            .reset_index(drop=True)
        )
        df.loc[result["__idx__"], new_col] = result[new_col].values

    def calculate_rate_based_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        calculate_rate_based_features calculates rate based features like the flows and bytes per source and destination IP address over a 1 second window.

        Args:
            df (pd.DataFrame): The input DataFrame to use for the calculations.

        Returns:
            pd.DataFrame: The output DataFrame with the engineered features added to it.
        """
        print("Calculating Rate-Based Features.")
        df = df.copy()

        apply_rolling(
            df, "Src IP", "Dst Port", "Flows_per_SrcIP_1s", lambda x: x.count()
        )
        apply_rolling(
            df, "Src IP", "Flow Bytes/s", "Bytes_per_SrcIP_1s", lambda x: x.sum()
        )
        apply_rolling(
            df, "Dst IP", "Src Port", "Flows_per_DstIP_1s", lambda x: x.count()
        )
        apply_rolling(
            df, "Dst IP", "Flow Bytes/s", "Bytes_per_DstIP_1s", lambda x: x.sum()
        )

        return df

    def calculate_uniqueness_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        calculate_uniqueness_features calculates engineered featured related to the number of unique destination ports per source port and vice versa.

        Args:
            df (pd.DataFrame): The input DataFrame to use for the calculations.

        Returns:
            pd.DataFrame: The DataFrame with the engineered features added to it.
        """
        print("Calculating Uniqueness Features.")
        df = df.copy()

        apply_rolling(
            df,
            "Src IP",
            "Dst Port",
            "Unique_DstPorts_per_SrcIP_1s",
            lambda x: x.nunique(),
        )
        apply_rolling(
            df, "Src IP", "Dst IP", "Unique_DstIPs_per_SrcIP_1s", lambda x: x.nunique()
        )
        apply_rolling(
            df, "Dst IP", "Src IP", "Unique_SrcIPs_per_DstIP_1s", lambda x: x.nunique()
        )
        apply_rolling(
            df,
            "Src IP",
            "Src Port",
            "Unique_SrcPorts_per_SrcIP_1s",
            lambda x: x.nunique(),
        )

        return df

    def calculate_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        calculate_behavioral_features calculates engineered features related to port and protocol numbers and average flow durations.

        Args:
            df (pd.DataFrame): The input DataFrame to use for the calculations.

        Returns:
            pd.DataFrame: The DataFrame with the engineered features added to it.
        """
        print("Calculating Behavioral Features...")
        df = df.copy()

        def entropy(x: pd.DataFrame) -> pd.DataFrame:
            """
            entropy predicts the number of Destination ports per Source port for a 1 second window.

            Args:
                x (pd.DataFrame): The input DataFrame to use for prediction.

            Returns:
                np.ndarray: The predicted number of Destination ports per Source Port for a 1 second window.
            """
            probs = x.value_counts(normalize=True)
            return -np.sum(p * np.log2(p) for p in probs if p > 0)

        apply_rolling(
            df, "Src IP", "Dst Port", "Entropy_DstPorts_per_SrcIP_1s", entropy
        )
        apply_rolling(
            df,
            "Src IP",
            "Flow Duration",
            "Avg_FlowDuration_per_SrcIP_1s",
            lambda x: x.mean(),
        )

        def map_port_category(port: int) -> str:
            """
            map_port_category converts numerical port numbers to categorical port names.

            Args:
                port (int): The port number.

            Returns:
                str: The corresponding port name.
            """
            port = int(port)
            if port in [80, 8080]:
                return "HTTP"
            elif port == 443:
                return "HTTPS"
            elif port == 22:
                return "SSH"
            elif port == 53:
                return "DNS"
            elif port in [25, 465, 587]:
                return "SMTP"
            elif port == 21:
                return "FTP"
            elif port in [3306, 5432]:
                return "DB"
            elif 0 <= port <= 1023:
                return "Well-Known"
            elif 49152 <= port <= 65535:
                return "Ephemeral"
            else:
                return "Other"

        df["DstPortCategory"] = df["Dst Port"].apply(map_port_category)
        df["SrcPortCategory"] = df["Src Port"].apply(map_port_category)

        def map_protocol_category(protocol: int) -> str:
            """
            map_protocol_category converts numerical protocol numbers to categorical protocol names.

            Args:
                protocol (int): The protocol number.

            Returns:
                str: The corresponding protocol name.
            """
            protocol = int(protocol)
            if protocol == 6:
                return "TCP"
            elif protocol == 17:
                return "UDP"
            elif protocol == 0:
                return "HOPOPT"
            else:
                raise Error(f"{protocol} is not a supported protocol number")

        df["ProtocolCategory"] = df["Protocol"].apply(map_protocol_category)

        return df

    df = prepare_dataframe(df)
    df = calculate_rate_based_features(df)
    df = calculate_uniqueness_features(df)
    df = calculate_behavioral_features(df)
    df.drop(columns="__idx__", inplace=True)

    # Drop the Timestamp column after using it for the time window calculations
    # Drop the Src Port, Dst Port, and Protocol columns after converting those values to a categorical representation
    df.drop(columns=["Timestamp", "Src Port", "Dst Port", "Protocol"], inplace=True)

    return df
