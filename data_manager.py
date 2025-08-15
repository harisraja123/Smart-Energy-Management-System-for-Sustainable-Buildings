import logging
from typing import List

import pandas as pd
import numpy as np
import os
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter('ignore', PerformanceWarning)


class DataManager:
    """DataManager handles data loading, preprocessing, feature engineering, and quality checks."""

    def __init__(self, timestamp_col: str = 'timestamp'):
        """
        Initialize the DataManager.

        Args:
            timestamp_col (str): Name of the timestamp column to parse and set as index.
        """
        self.timestamp_col = timestamp_col
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from a CSV file and set timestamp index.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataframe with datetime index.
        """
        # Read raw CSV first
        df = pd.read_csv(file_path)
        # If timestamp column exists, parse as datetime and set as index
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df.set_index(self.timestamp_col, inplace=True)
        else:
            # No timestamp column: keep default integer index
            df.reset_index(drop=True, inplace=True)
        return df

    def integrate_datasets(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and concatenate multiple CSV datasets.

        Args:
            file_paths (List[str]): List of CSV file paths.

        Returns:
            pd.DataFrame: Concatenated dataframe sorted by timestamp.
        """
        dfs = []
        for fp in file_paths:
            df = self.load_csv(fp)
            dfs.append(df)
        df = pd.concat(dfs).sort_index()
        return df

    def quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data quality checks: range validation, anomaly detection, temporal consistency.

        Range validation uses sensor specifications:
            - temperature: -10°C to 50°C
            - humidity: 0% to 100%

        Anomalies are values more than 3 standard deviations from the mean.

        Args:
            df (pd.DataFrame): Input dataframe with raw sensor data.

        Returns:
            pd.DataFrame: Dataframe with anomaly flags added and sorted index.
        """
        issues = []
        # Range validation for all temperature-like columns
        temp_cols = [c for c in df.columns if 'temperature' in c.lower()]
        for col in temp_cols:
            invalid = df[(df[col] < -10) | (df[col] > 50)]
            if not invalid.empty:
                self.logger.debug(f"{col} values out of range: {len(invalid)} rows.")
                issues.append(f"{col}_range")
        # Range validation for all humidity-like columns
        hum_cols = [c for c in df.columns if 'humidity' in c.lower()]
        for col in hum_cols:
            invalid = df[(df[col] < 0) | (df[col] > 100)]
            if not invalid.empty:
                self.logger.debug(f"{col} values out of range: {len(invalid)} rows.")
                issues.append(f"{col}_range")

        # Anomaly detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            anomaly_mask = (df[col] - mean).abs() > 3 * std
            if anomaly_mask.any():
                count = anomaly_mask.sum()
                self.logger.debug(f"Anomalies detected in {col}: {count} rows.")
                df[f'{col}_anomaly'] = 0
                df.loc[anomaly_mask, f'{col}_anomaly'] = 1
                issues.append(f'{col}_anomaly')

        # Temporal consistency: ensure sorted by timestamp
        if not df.index.is_monotonic_increasing:
            self.logger.warning("Timestamps are not ordered. Sorting index.")
            df.sort_index(inplace=True)
            issues.append('timestamp_order')

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform preprocessing pipeline: missing value handling, normalization,
        feature engineering, and time-based feature creation.

        Steps:
            1. Forward- and backward-fill missing values.
            2. Normalize temperature to [0,1] using range -10°C to 50°C.
            3. Binary encode occupancy.
            4. Create rolling averages for temperature and humidity over
               1h, 6h, and 24h windows.
            5. Add time-based features: hour, day_of_week, season.

        Args:
            df (pd.DataFrame): Input dataframe after quality checks.

        Returns:
            pd.DataFrame: Preprocessed dataframe with new features.
        """
        # Sort index for rolling computations
        df = df.sort_index()

        # Missing values handling
        df = df.ffill().bfill()
        # Normalize all temperature-like columns
        temp_cols = [c for c in df.columns if 'temperature' in c.lower()]
        for col in temp_cols:
            df[col] = (df[col] + 10) / 60.0
        # Binary encode all occupancy-like columns
        occ_cols = [c for c in df.columns if 'occupancy' in c.lower() or 'occupant' in c.lower()]
        for col in occ_cols:
            df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
        # Compute rolling averages for temperature and humidity columns
        windows_offset = {'1h': '1H', '6h': '6H', '24h': '24H'}
        windows_periods = {label: int(label.rstrip('h')) for label in windows_offset}
        hum_cols = [c for c in df.columns if 'humidity' in c.lower()]
        rolling_data = {}
        for label in windows_offset:
            if isinstance(df.index, pd.DatetimeIndex):
                window = windows_offset[label]
            else:
                window = windows_periods[label]
            for col in temp_cols + hum_cols:
                rolling_data[f'{col}_mean_{label}'] = df[col].rolling(window=window, min_periods=1).mean()
        # Concatenate rolling features in one go to avoid fragmentation
        rolling_df = pd.DataFrame(rolling_data, index=df.index)
        df = pd.concat([df, rolling_df], axis=1)
        # Time-based features only when index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['season'] = df['month'].apply(self._map_season)
            df.drop(columns=['month'], inplace=True)
        # Return a defragmented copy to eliminate fragmentation warnings
        return df.copy()

    def _map_season(self, month: int) -> str:
        """
        Map month to season string.

        Args:
            month (int): Month as integer (1–12).

        Returns:
            str: Season ('winter', 'spring', 'summer', 'autumn').
        """
        if month in [12, 1, 2]:
            return 'winter'
        if month in [3, 4, 5]:
            return 'spring'
        if month in [6, 7, 8]:
            return 'summer'
        return 'autumn'

    def load_dataset(self, dataset_dir: str) -> pd.DataFrame:
        """
        Discover all CSV files in `dataset_dir` (e.g. building_*.csv, weather.csv,
        carbon_intensity.csv), prefix each DataFrame’s columns by filename,
        horizontally merge them on timestamp, then run quality checks and preprocessing.

        Args:
            dataset_dir (str): Directory containing the CSV files.

        Returns:
            pd.DataFrame: Fully merged and preprocessed dataset ready for RL.
        """
        # Gather CSV files
        file_names = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        file_paths = sorted(os.path.join(dataset_dir, f) for f in file_names)

        # Load and prefix each CSV
        dfs = []
        for fp in file_paths:
            df = self.load_csv(fp)
            prefix = os.path.splitext(os.path.basename(fp))[0]
            df = df.add_prefix(prefix + '_')
            dfs.append(df)

        # Merge all features by timestamp
        merged = pd.concat(dfs, axis=1).sort_index()

        # Run quality checks and preprocessing
        merged = self.quality_check(merged)
        merged = self.preprocess(merged)
        return merged

    def get_available_buildings(self, data_dir: str) -> List[int]:
        """
        Get list of available building IDs from the data directory.

        Args:
            data_dir (str): Path to the data directory

        Returns:
            List[int]: List of available building IDs
        """
        try:
            # Check for building CSV files
            import glob
            building_files = glob.glob(os.path.join(data_dir, 'Building_*.csv'))

            # Extract building numbers from filenames
            available_buildings = []
            for f in building_files:
                try:
                    building_id = int(os.path.basename(f).split('_')[1].split('.')[0])
                    available_buildings.append(building_id)
                except (IndexError, ValueError):
                    self.logger.warning(f"Could not parse building ID from filename: {f}")
                    continue

            return sorted(available_buildings)

        except Exception as e:
            self.logger.error(f"Error getting available buildings: {str(e)}")
            raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Data Collection and Preprocessing Pipeline for Smart Energy RL'
    )
    parser.add_argument(
        '--input_files',
        nargs='+',
        required=True,
        help='List of CSV files containing raw sensor data'
    )
    parser.add_argument(
        '--output_file',
        required=True,
        help='Path to save processed data CSV'
    )
    args = parser.parse_args()

    manager = DataManager(timestamp_col='timestamp')
    raw_df = manager.integrate_datasets(args.input_files)
    checked_df = manager.quality_check(raw_df)
    processed_df = manager.preprocess(checked_df)
    processed_df.to_csv(args.output_file)