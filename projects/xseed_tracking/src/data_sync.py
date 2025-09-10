import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, time
import logging
import traceback
class DataSynchronizer:
    def __init__(self):
        self.standard_frequency = '100ms'
        self.max_gap_seconds = 1.0
        self.min_match_rate = 0.8

    def standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        standardized_df = df.copy()
        
        timestamp_cols = ['timestamp', 'gps_time', 'time', 'date time']
        time_col = next((col for col in timestamp_cols if col in df.columns), None)
        
        if not time_col:
            raise ValueError("No timestamp column found")
            
        try:
            standardized_df['date time'] = pd.to_datetime(
                standardized_df[time_col]
            )
            standardized_df['time_of_day'] = standardized_df['date time'].dt.time
            standardized_df['seconds'] = standardized_df['time_of_day'].apply(
                lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond/1e6
            )
            return standardized_df
        except Exception as e:
            logging.error(f"Error standardizing timestamps: {str(e)}")
            raise

    def downsample_gpexe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsample GPS data to exactly 10Hz using precise time-based resampling.
        
        Args:
            df: DataFrame to downsample
        """
        try:
            if df.empty:
                logging.warning("Cannot downsample empty DataFrame")
                return df
                
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure date time column exists and is properly formatted
            if 'date time' not in df.columns:
                logging.error("No 'date time' column found for downsampling")
                return df
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df['date time']):
                df['date time'] = pd.to_datetime(df['date time'], errors='coerce')
                # Drop rows with invalid timestamps
                df = df.dropna(subset=['date time'])
                
            # Calculate original frequency for logging
            time_span = (df['date time'].max() - df['date time'].min()).total_seconds()
            original_points = len(df)
            original_hz = original_points / time_span if time_span > 0 else 0
            logging.info(f"Original GPS data: {original_points} points over {time_span:.2f}s ({original_hz:.2f} Hz)")
            
            # Sort by timestamp
            df = df.sort_values('date time')
            
            # Set datetime as index
            df = df.set_index('date time')
            
            # Resample to exactly 10Hz (100ms)
            df_resampled = df.resample('100ms').asfreq()
            
            # Interpolate missing values
            df_resampled = df_resampled.interpolate(
                method='linear',
                limit_direction='both',
                limit=5  # Maximum 5 points interpolation
            )
            
            # Reset index
            df_resampled = df_resampled.reset_index()
            
            # Verify output frequency
            time_span = (df_resampled['date time'].max() - df_resampled['date time'].min()).total_seconds()
            actual_hz = len(df_resampled) / time_span if time_span > 0 else 0
            logging.info(f"Resampled GPS data: {len(df_resampled)} points over {time_span:.2f}s ({actual_hz:.2f} Hz)")
            
            return df_resampled
            
        except Exception as e:
            logging.error(f"Error in downsample_gpexe: {str(e)}")
            logging.error(traceback.format_exc())
            return df

    def synchronize_streams(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          time_tolerance: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            
            base_date = datetime.now().date()  # Get current date for time alignment
            df1 = self.standardize_timestamps(df1.copy())
            df2 = self.standardize_timestamps(df2.copy())
            
            start_secs = max(df1['seconds'].min(), df2['seconds'].min())
            end_secs = min(df1['seconds'].max(), df2['seconds'].max())
            
            start_with_tolerance = start_secs - time_tolerance
            end_with_tolerance = end_secs + time_tolerance
            
            mask1 = (df1['seconds'] >= start_with_tolerance) & \
        (df1['seconds'] <= end_with_tolerance)
            mask2 = (df2['seconds'] >= start_with_tolerance) & \
                    (df2['seconds'] <= end_with_tolerance)
            
            return df1[mask1].copy(), df2[mask2].copy()
            
        except Exception as e:
            logging.error(f"Error synchronizing streams: {str(e)}")
            raise
    def merge_synchronized_data(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              tolerance: float = 0.1) -> pd.DataFrame:
        try:
            merged = pd.merge_asof(
                df1.sort_values('seconds'),
                df2.sort_values('seconds'),
                on='seconds',
                tolerance=tolerance,
                suffixes=('_1', '_2')
            )
            
            return merged
            
        except Exception as e:
            logging.error(f"Error merging synchronized data: {str(e)}")
            return pd.DataFrame()
        
    def interpolate_missing_data(self, df: pd.DataFrame, 
                               max_gap_seconds: Optional[float] = None,
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Interpolate missing data points.
        
        Args:
            df: DataFrame with missing data
            max_gap_seconds: Maximum gap to interpolate
            columns: Specific columns to interpolate
        """
        try:
            max_gap = max_gap_seconds or self.max_gap_seconds
            interpolated_df = df.copy()
            
            # Set datetime as index
            interpolated_df = interpolated_df.set_index('date time')
            
            # Calculate time differences
            time_diff = interpolated_df.index.to_series().diff()
            
            # Find gaps smaller than max_gap
            small_gaps = time_diff <= pd.Timedelta(seconds=max_gap)
            
            # Select columns to interpolate
            if columns is None:
                columns = interpolated_df.select_dtypes(include=[np.number]).columns
            else:
                columns = [col for col in columns if col in interpolated_df.columns]
            
            # Interpolate selected columns
            for col in columns:
                interpolated_df[col] = interpolated_df[col].interpolate(
                    method='time',
                    limit_area='inside',
                    limit=int(max_gap / float(self.standard_frequency[:-2]) * 1000)
                )
            
            return interpolated_df.reset_index()
            
        except Exception as e:
            logging.error(f"Error interpolating data: {str(e)}")
            return df

    def validate_synchronization(self, merged_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate synchronization quality.
        
        Args:
            merged_data: Synchronized DataFrame
        """
        validation = {
            'complete_timestamps': False,
            'acceptable_gaps': False,
            'consistent_sampling': False,
            'matched_events': False
        }
        
        try:
            # Check timestamp completeness
            validation['complete_timestamps'] = not merged_data['date time'].isnull().any()
            
            # Check time gaps
            time_gaps = merged_data['date time'].diff().dt.total_seconds()
            validation['acceptable_gaps'] = time_gaps.max() <= self.max_gap_seconds
            
            # Check sampling consistency
            mean_gap = time_gaps.mean()
            std_gap = time_gaps.std()
            validation['consistent_sampling'] = (std_gap / mean_gap) < 0.5
            
            # Check event matching if available
            if 'event_1' in merged_data.columns and 'event_2' in merged_data.columns:
                validation['matched_events'] = (
                    merged_data['event_1'] == merged_data['event_2']
                ).mean() > self.min_match_rate
            
            return validation
            
        except Exception as e:
            logging.error(f"Error validating synchronization: {str(e)}")
            return validation

    def detect_sync_errors(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect synchronization errors in data.
        
        Args:
            df: DataFrame to check for errors
        """
        errors = []
        
        try:
            # Calculate time differences
            time_diffs = df['date time'].diff().dt.total_seconds()
            
            # Detect large gaps
            large_gaps = time_diffs[time_diffs > self.max_gap_seconds]
            
            for idx, gap in large_gaps.items():
                errors.append({
                    'type': 'time_gap',
                    'index': idx,
                    'timestamp': df['date time'].iloc[idx],
                    'gap_size': gap,
                    'severity': 'high' if gap > self.max_gap_seconds * 2 else 'medium'
                })
            
            # Check for misaligned events if available
            if all(col in df.columns for col in ['event_1', 'event_2']):
                misaligned = df[df['event_1'] != df['event_2']]
                for idx, row in misaligned.iterrows():
                    errors.append({
                        'type': 'misaligned_events',
                        'index': idx,
                        'timestamp': row['date time'],
                        'event_1': row['event_1'],
                        'event_2': row['event_2']
                    })
            
            return errors
            
        except Exception as e:
            logging.error(f"Error detecting sync errors: {str(e)}")
            return []

    def calculate_sync_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate synchronization quality metrics.
        
        Args:
            df: DataFrame to analyze
        """
        try:
            # Initialize metrics
            metrics = {
                'completeness': 0.0,
                'temporal_consistency': 0.0,
                'alignment_score': 0.0,
                'overall_quality': 0.0
            }
            
            if df.empty:
                return metrics
            
            # Calculate completeness
            time_span = (df['date time'].max() - df['date time'].min()).total_seconds()
            expected_points = time_span * 10  # For 10Hz sampling
            metrics['completeness'] = min(1.0, len(df) / expected_points)
            
            # Calculate temporal consistency
            time_diffs = df['date time'].diff().dt.total_seconds()
            metrics['temporal_consistency'] = 1.0 - min(1.0, time_diffs.std() / time_diffs.mean())
            
            # Calculate overall quality
            metrics['overall_quality'] = np.mean([
                metrics['completeness'],
                metrics['temporal_consistency']
            ])
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating sync quality metrics: {str(e)}")
            return metrics

    def resample_data(self, df: pd.DataFrame, target_frequency: str) -> pd.DataFrame:
        """
        Resample data to target frequency with enhanced error handling.
        
        Args:
            df: DataFrame to resample
            target_frequency: Target sampling frequency (e.g., '100ms' for 10Hz)
            
        Returns:
            Resampled DataFrame
        """
        try:
            if df.empty:
                logging.warning("Cannot resample empty DataFrame")
                return df
                
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure date time column exists and is properly formatted
            if 'date time' not in df_copy.columns:
                logging.error("No 'date time' column found for resampling")
                return df
                
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(df_copy['date time']):
                df_copy['date time'] = pd.to_datetime(df_copy['date time'], errors='coerce')
                # Drop rows with invalid timestamps
                df_copy = df_copy.dropna(subset=['date time'])
                
            # Sort by timestamp
            df_copy = df_copy.sort_values('date time')
            
            # Calculate original frequency for logging
            time_diffs = df_copy['date time'].diff().dropna()
            if not time_diffs.empty:
                mean_diff = time_diffs.mean().total_seconds()
                orig_freq = 1.0 / mean_diff if mean_diff > 0 else 0
                logging.info(f"Original data frequency: {orig_freq:.2f} Hz")
                
            # Set datetime as index
            df_indexed = df_copy.set_index('date time')
            
            # Resample data
            resampled = df_indexed.resample(target_frequency).asfreq()
            
            # Interpolate numeric columns
            numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                resampled[col] = resampled[col].interpolate(
                    method='time',
                    limit=int(self.max_gap_seconds / float(target_frequency[:-2]) * 1000)
                )
            
            # Reset index to get date time as column again
            result = resampled.reset_index()
            
            # Log resampling results
            target_hz = 1000 / float(target_frequency[:-2]) if target_frequency.endswith('ms') else 1.0
            logging.info(f"Resampled from {len(df)} to {len(result)} points (target: {target_hz:.2f} Hz)")
            
            return result
            
        except Exception as e:
            logging.error(f"Error resampling data: {str(e)}")
            logging.error(traceback.format_exc())
            return df

    def correct_timestamp_drift(self, df: pd.DataFrame, 
                              reference_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Correct timestamp drift.
        
        Args:
            df: DataFrame with potentially drifting timestamps
            reference_time: Optional reference timestamp
        """
        try:
            corrected_df = df.copy()
            
            if reference_time is None:
                reference_time = corrected_df['date time'].iloc[0]
            
            # Calculate time differences
            time_diffs = corrected_df['date time'].diff()
            
            # Detect and correct drift
            drift_threshold = pd.Timedelta(milliseconds=50)
            cumulative_drift = pd.Timedelta(0)
            
            for i in range(1, len(corrected_df)):
                current_diff = time_diffs.iloc[i]
                if abs(current_diff) > drift_threshold:
                    cumulative_drift += (current_diff - drift_threshold)
                    corrected_df.loc[i:, 'date time'] -= cumulative_drift
            
            return corrected_df
            
        except Exception as e:
            logging.error(f"Error correcting timestamp drift: {str(e)}")
            return df