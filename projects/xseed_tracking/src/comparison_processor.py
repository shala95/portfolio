import pandas as pd
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import traceback
from datetime import datetime, timedelta
from src.data_sync import DataSynchronizer 
class ComparisonProcessor:

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.data_sync = DataSynchronizer()
    def _empty_metrics(self) -> Dict:
        """Return empty basic metrics dictionary."""
        return {
            'mean_error': 0.0,
            'rmse': 0.0,
            'min_error': 0.0,
            'max_error': 0.0,
            'median_error': 0.0,
            'matching_rate': 0.0,
            'out_of_bounds_count': 0,
            'out_of_bounds_mean': 0.0,
            'error_distribution': [],
            'gt_points_used': 0,
            'gt_points_usage_pct': 0.0,
            'avg_matches_per_used_gt': 0.0,
            'gt_points_total': 0,
            'skipped_time_window': 0,
            'skipped_future_progression': 0,
            'skipped_max_matches': 0
        }
    def calculate_tracking_accuracy(self, ground_truth: pd.DataFrame,
                  tracking_data: pd.DataFrame,
                  time_tolerance_seconds: int = 2,
                  max_matches_per_gt_point: int = 3,
                  ignore_date: bool = True) -> Dict[str, float]:
        """
        Calculate tracking accuracy metrics comparing ground truth with tracking data.
        
        Enhanced version with time window filtering, match limiting, forward-time progression
        constraint, and vectorized calculations. Added detailed debug logging.
        
        Args:
            ground_truth: DataFrame with ground truth positions
            tracking_data: DataFrame with tracking system positions
            time_tolerance_seconds: Time window size in seconds for matching points
            max_matches_per_gt_point: Maximum times a ground truth point can be matched
            ignore_date: Whether to ignore the date part of timestamps and match only time-of-day
            
        Returns:
            Dictionary with error metrics
        """
        if ground_truth.empty or tracking_data.empty:
            logging.warning("Empty data provided: ground_truth empty = {}, tracking_data empty = {}"
                        .format(ground_truth.empty, tracking_data.empty))
            return self._empty_metrics()

        try:
            # Add detailed debugging logs
            logging.info("----- DETAILED DEBUG INFORMATION -----")
            
            # Log timestamp ranges
            gt_min_time = ground_truth['date time'].min() if 'date time' in ground_truth.columns else None
            gt_max_time = ground_truth['date time'].max() if 'date time' in ground_truth.columns else None
            track_min_time = tracking_data['date time'].min() if 'date time' in tracking_data.columns else None
            track_max_time = tracking_data['date time'].max() if 'date time' in tracking_data.columns else None
            
            logging.info(f"Ground Truth Time Range: {gt_min_time} to {gt_max_time}")
            logging.info(f"Tracking Data Time Range: {track_min_time} to {track_max_time}")
            
            # Log coordinate ranges
            if 'x_real_m' in ground_truth.columns and 'y_real_m' in ground_truth.columns:
                gt_x_min, gt_x_max = ground_truth['x_real_m'].min(), ground_truth['x_real_m'].max()
                gt_y_min, gt_y_max = ground_truth['y_real_m'].min(), ground_truth['y_real_m'].max()
                logging.info(f"Ground Truth X Range: {gt_x_min:.2f} to {gt_x_max:.2f}")
                logging.info(f"Ground Truth Y Range: {gt_y_min:.2f} to {gt_y_max:.2f}")
            else:
                logging.warning("Ground Truth missing x_real_m or y_real_m columns")
                logging.info(f"Available Ground Truth columns: {ground_truth.columns.tolist()}")
                
            if 'x_real_m' in tracking_data.columns and 'y_real_m' in tracking_data.columns:
                track_x_min, track_x_max = tracking_data['x_real_m'].min(), tracking_data['x_real_m'].max()
                track_y_min, track_y_max = tracking_data['y_real_m'].min(), tracking_data['y_real_m'].max()
                logging.info(f"Tracking Data X Range: {track_x_min:.2f} to {track_x_max:.2f}")
                logging.info(f"Tracking Data Y Range: {track_y_min:.2f} to {track_y_max:.2f}")
            else:
                logging.warning("Tracking Data missing x_real_m or y_real_m columns")
                logging.info(f"Available Tracking Data columns: {tracking_data.columns.tolist()}")
                
                # Try to find alternative coordinate columns
                x_candidates = [col for col in tracking_data.columns if 'x' in col.lower()]
                y_candidates = [col for col in tracking_data.columns if 'y' in col.lower()]
                if x_candidates and y_candidates:
                    logging.info(f"Potential X columns: {x_candidates}")
                    logging.info(f"Potential Y columns: {y_candidates}")
                    
                    # Sample values from first alternative coordinate columns
                    alt_x_col = x_candidates[0] 
                    alt_y_col = y_candidates[0]
                    logging.info(f"Sample values from '{alt_x_col}': {tracking_data[alt_x_col].head(3).tolist()}")
                    logging.info(f"Sample values from '{alt_y_col}': {tracking_data[alt_y_col].head(3).tolist()}")
            
            # Log data samples
            logging.info(f"Ground Truth sample (first 3 rows):")
            for idx, row in ground_truth.head(3).iterrows():
                timestamp = row.get('date time', 'N/A')
                x = row.get('x_real_m', 'N/A')
                y = row.get('y_real_m', 'N/A')
                logging.info(f"  Row {idx}: timestamp={timestamp}, x={x}, y={y}")
                
            logging.info(f"Tracking Data sample (first 3 rows):")
            for idx, row in tracking_data.head(3).iterrows():
                timestamp = row.get('date time', 'N/A')
                x = row.get('x_real_m', 'N/A')
                y = row.get('y_real_m', 'N/A')
                logging.info(f"  Row {idx}: timestamp={timestamp}, x={x}, y={y}")
                
            logging.info(f"Time tolerance set to {time_tolerance_seconds} seconds")
            logging.info("----- END DEBUG INFORMATION -----")
            
            max_distance_threshold = 5.0
            point_errors = []
            out_of_bounds_errors = []
            gt_match_counts = np.zeros(len(ground_truth), dtype=int)  # Track matches per GT point
            
            # Check if we have interpolation information
            has_interp_info = 'data_quality' in tracking_data.columns
            if has_interp_info:
                original_errors = []
                interpolated_errors = []
            
            # Create numpy arrays of coordinates for vectorized calculations
            try:
                gt_x = ground_truth['x_real_m'].to_numpy()
                gt_y = ground_truth['y_real_m'].to_numpy()
                logging.info(f"Ground Truth coordinate arrays created with shapes: x={gt_x.shape}, y={gt_y.shape}")
            except Exception as e:
                logging.error(f"Error creating ground truth coordinate arrays: {str(e)}")
                return self._empty_metrics()
            
            # Convert timestamps to numpy arrays with proper handling of dates
            if ignore_date:
                logging.info(f"Using time-of-day comparison (ignoring dates)")
                
                # Convert timestamps to seconds-of-day
                gt_seconds_of_day = np.array([
                    ts.hour * 3600 + ts.minute * 60 + ts.second + (ts.microsecond/1e6 if hasattr(ts, 'microsecond') else 0)
                    for ts in ground_truth['date time']
                ])
                
                track_seconds_of_day = np.array([
                    ts.hour * 3600 + ts.minute * 60 + ts.second + (ts.microsecond/1e6 if hasattr(ts, 'microsecond') else 0)
                    for ts in tracking_data['date time']
                ])
                
                # Use seconds-of-day for comparison
                gt_ts = gt_seconds_of_day
                track_timestamps = track_seconds_of_day
                
                # Get indices that would sort tracking data by time-of-day
                track_sorted_indices = np.argsort(track_seconds_of_day)
            else:
                # Use full timestamps for comparison
                try:
                    gt_ts = np.array([ts.timestamp() for ts in ground_truth['date time']])
                    track_timestamps = np.array([ts.timestamp() for ts in tracking_data['date time']])
                    logging.info(f"Converted {len(gt_ts)} ground truth timestamps and {len(track_timestamps)} tracking timestamps")
                except Exception as e:
                    logging.error(f"Error converting timestamps: {str(e)}")
                    logging.info(f"Ground Truth date time dtype: {ground_truth['date time'].dtype}")
                    logging.info(f"Tracking date time dtype: {tracking_data['date time'].dtype}")
                    # Try to convert timestamp samples
                    logging.info("Trying to convert sample timestamps:")
                    try:
                        gt_sample = ground_truth['date time'].iloc[0]
                        track_sample = tracking_data['date time'].iloc[0]
                        logging.info(f"Ground Truth sample: {gt_sample}, type: {type(gt_sample)}")
                        logging.info(f"Tracking sample: {track_sample}, type: {type(track_sample)}")
                        logging.info(f"Ground Truth to timestamp: {gt_sample.timestamp() if hasattr(gt_sample, 'timestamp') else 'N/A'}")
                        logging.info(f"Tracking to timestamp: {track_sample.timestamp() if hasattr(track_sample, 'timestamp') else 'N/A'}")
                    except Exception as se:
                        logging.error(f"Sample timestamp conversion error: {str(se)}")
                    
                    # Fall back to safer conversion
                    logging.info("Falling back to slower timestamp conversion method")
                    gt_ts = np.array([pd.Timestamp(ts).timestamp() if pd.notna(ts) else np.nan for ts in ground_truth['date time']])
                    track_timestamps = np.array([pd.Timestamp(ts).timestamp() if pd.notna(ts) else np.nan for ts in tracking_data['date time']])
                
                # Get indices that would sort tracking data by time
                track_sorted_indices = np.argsort(track_timestamps)
            
            logging.info(f"Processing {len(track_sorted_indices)} tracking points in time order")
            
            # Stats for skipped points
            skipped_time_window_points = 0
            skipped_future_points = 0
            skipped_max_matches_points = 0
            
            # Track the last matched GT timestamp to enforce forward progression
            last_matched_gt_timestamp = None
            total_matches_attempted = 0
            
            # Process each tracking point in time order
            for sorted_idx in track_sorted_indices:
                track_point = tracking_data.iloc[sorted_idx]
                
                # Get timestamp value based on ignore_date setting
                if ignore_date:
                    track_point_time = track_seconds_of_day[sorted_idx]
                else:
                    track_point_time = track_point['date time'].timestamp()
                    
                total_matches_attempted += 1
                
                if total_matches_attempted == 1 or total_matches_attempted % 50 == 0 or total_matches_attempted == len(track_sorted_indices):
                    logging.info(f"Processing point {total_matches_attempted}/{len(track_sorted_indices)}, timestamp: {track_point['date time']}")
                
                # Calculate time differences and find points within tolerance window
                time_diffs = np.abs(gt_ts - track_point_time)
                within_time_window = time_diffs <= time_tolerance_seconds
                
                points_in_window = np.sum(within_time_window)
                if total_matches_attempted == 1:
                    logging.info(f"First point has {points_in_window} ground truth points within time window ({time_tolerance_seconds}s)")
                
                # Skip if no points are within time window
                if not np.any(within_time_window):
                    skipped_time_window_points += 1
                    if skipped_time_window_points == 1 or skipped_time_window_points % 50 == 0:
                        logging.info(f"Point at {track_point['date time']} skipped: No points within time window")
                        # Sample time difference to nearest GT point
                        min_diff_idx = np.argmin(time_diffs)
                        min_diff = time_diffs[min_diff_idx]
                        nearest_gt_time = ground_truth['date time'].iloc[min_diff_idx]
                        logging.info(f"  Nearest GT point is {min_diff:.2f}s away at {nearest_gt_time}")
                    continue
                
                # Apply forward time progression constraint
                if last_matched_gt_timestamp is not None:
                    # Only consider GT points with timestamps >= last matched GT timestamp
                    future_constraint = gt_ts >= last_matched_gt_timestamp
                    # Combine with time window constraint
                    within_constraints = within_time_window & future_constraint
                    
                    # Skip if no points meet both constraints
                    if not np.any(within_constraints):
                        skipped_future_points += 1
                        if skipped_future_points == 1:
                            logging.info(f"Point at {track_point['date time']} skipped: No future points available")
                            if ignore_date:
                                logging.info(f"  Last matched GT seconds-of-day: {last_matched_gt_timestamp:.2f}")
                            else:
                                logging.info(f"  Last matched GT timestamp: {datetime.fromtimestamp(last_matched_gt_timestamp)}")
                            # Log which constraint failed
                            future_points = np.sum(future_constraint)
                            logging.info(f"  Points in future: {future_points}, Points in time window: {points_in_window}")
                        continue
                        
                    candidate_indices = np.where(within_constraints)[0]
                else:
                    # For the first match, just use time window constraint
                    candidate_indices = np.where(within_time_window)[0]
                
                # Filter out ground truth points that have reached max matches
                available_indices = [i for i in candidate_indices if gt_match_counts[i] < max_matches_per_gt_point]
                
                if len(available_indices) == 0:
                    skipped_max_matches_points += 1
                    if skipped_max_matches_points == 1:
                        logging.info(f"Point at {track_point['date time']} skipped: All candidate points reached max matches")
                        logging.info(f"  Candidate points: {len(candidate_indices)}")
                    continue
                    
                # Calculate distances to ground truth points within constraints
                x_diffs = gt_x[available_indices] - track_point['x_real_m']
                y_diffs = gt_y[available_indices] - track_point['y_real_m']
                distances = np.sqrt(x_diffs**2 + y_diffs**2)
                
                # Find minimum distance
                if len(distances) > 0:
                    min_dist_idx = np.argmin(distances)
                    min_error = distances[min_dist_idx]
                    gt_idx = available_indices[min_dist_idx]
                    
                    # Log the first successful match details
                    if len(point_errors) == 0:
                        gt_point = ground_truth.iloc[gt_idx]
                        logging.info(f"First successful match:")
                        logging.info(f"  Track point: time={track_point['date time']}, x={track_point['x_real_m']:.2f}, y={track_point['y_real_m']:.2f}")
                        logging.info(f"  GT point: time={gt_point['date time']}, x={gt_point['x_real_m']:.2f}, y={gt_point['y_real_m']:.2f}")
                        logging.info(f"  Error: {min_error:.2f}m, Time diff: {abs((gt_point['date time'] - track_point['date time']).total_seconds()):.2f}s")
                    
                    # Check if error is within threshold
                    if min_error <= max_distance_threshold:
                        point_errors.append(min_error)
                        gt_match_counts[gt_idx] += 1  # Increment match count for this GT point
                        
                        # Update last matched GT timestamp
                        last_matched_gt_timestamp = gt_ts[gt_idx]
                        
                        # Every 50 successful matches, log some stats
                        if len(point_errors) % 50 == 0:
                            logging.info(f"Successful matches so far: {len(point_errors)}")
                            logging.info(f"Mean error so far: {np.mean(point_errors):.2f}m")
                        
                        # Store in quality-specific arrays if we have that info
                        if has_interp_info:
                            if track_point.get('data_quality') == 'interpolated':
                                interpolated_errors.append(min_error)
                            else:
                                original_errors.append(min_error)
                    else:
                        out_of_bounds_errors.append(min_error)
                        if len(out_of_bounds_errors) == 1:
                            logging.info(f"First out-of-bounds error: {min_error:.2f}m (threshold: {max_distance_threshold}m)")
                            gt_point = ground_truth.iloc[gt_idx]
                            logging.info(f"  Track point: time={track_point['date time']}, x={track_point['x_real_m']:.2f}, y={track_point['y_real_m']:.2f}")
                            logging.info(f"  GT point: time={gt_point['date time']}, x={gt_point['x_real_m']:.2f}, y={gt_point['y_real_m']:.2f}")
            
            # Final processing logs
            logging.info(f"Matching complete. Successful matches: {len(point_errors)}")
            logging.info(f"Skipped due to time window: {skipped_time_window_points}")
            logging.info(f"Skipped due to future constraint: {skipped_future_points}")
            logging.info(f"Skipped due to max matches: {skipped_max_matches_points}")
            logging.info(f"GT points used: {np.sum(gt_match_counts > 0)}/{len(ground_truth)}")
            
            # Convert to numpy arrays for faster calculations
            valid_errors = np.array(point_errors) if point_errors else np.array([])
            invalid_errors = np.array(out_of_bounds_errors) if out_of_bounds_errors else np.array([])
            
            # Calculate GT points usage statistics
            gt_points_used = np.sum(gt_match_counts > 0)
            gt_points_usage_pct = (gt_points_used / len(ground_truth)) * 100 if len(ground_truth) > 0 else 0
            avg_matches_per_used_gt = np.mean(gt_match_counts[gt_match_counts > 0]) if gt_points_used > 0 else 0
            
            logging.info(f"Calculation complete. Valid errors: {len(valid_errors)}, Invalid errors: {len(invalid_errors)}")
            
            # Calculate base metrics
            metrics = {
                'mean_error': float(np.mean(valid_errors)) if len(valid_errors) > 0 else float('inf'),
                'rmse': float(np.sqrt(np.mean(valid_errors**2))) if len(valid_errors) > 0 else float('inf'),
                'min_error': float(np.min(valid_errors)) if len(valid_errors) > 0 else float('inf'),
                'max_error': float(np.max(valid_errors)) if len(valid_errors) > 0 else float('inf'),
                'median_error': float(np.median(valid_errors)) if len(valid_errors) > 0 else float('inf'),
                'matching_rate': len(valid_errors) / len(tracking_data) if len(tracking_data) > 0 else 0.0,
                'out_of_bounds_count': len(invalid_errors),
                'out_of_bounds_mean': float(np.mean(invalid_errors)) if len(invalid_errors) > 0 else 0.0,
                'error_distribution': valid_errors.tolist(),
                # Add GT point usage statistics
                'gt_points_used': int(gt_points_used),
                'gt_points_usage_pct': float(gt_points_usage_pct),
                'avg_matches_per_used_gt': float(avg_matches_per_used_gt),
                'gt_points_total': len(ground_truth),
                # Add skipped points statistics
                'skipped_time_window': skipped_time_window_points,
                'skipped_future_progression': skipped_future_points,
                'skipped_max_matches': skipped_max_matches_points
            }
            
            # Add enhanced metrics for interpolated data if available
            if has_interp_info:
                orig_errors_np = np.array(original_errors) if original_errors else np.array([])
                interp_errors_np = np.array(interpolated_errors) if interpolated_errors else np.array([])
                
                metrics['has_interp_data'] = True
                metrics['original'] = {
                    'mean_error': float(np.mean(orig_errors_np)) if len(orig_errors_np) > 0 else float('inf'),
                    'rmse': float(np.sqrt(np.mean(orig_errors_np**2))) if len(orig_errors_np) > 0 else float('inf'),
                    'min_error': float(np.min(orig_errors_np)) if len(orig_errors_np) > 0 else float('inf'),
                    'max_error': float(np.max(orig_errors_np)) if len(orig_errors_np) > 0 else float('inf'),
                    'median_error': float(np.median(orig_errors_np)) if len(orig_errors_np) > 0 else float('inf'),
                    'count': len(orig_errors_np),
                    'percentage': 100 * len(orig_errors_np) / len(valid_errors) if len(valid_errors) > 0 else 0.0
                }
                
                metrics['interpolated'] = {
                    'mean_error': float(np.mean(interp_errors_np)) if len(interp_errors_np) > 0 else float('inf'),
                    'rmse': float(np.sqrt(np.mean(interp_errors_np**2))) if len(interp_errors_np) > 0 else float('inf'),
                    'min_error': float(np.min(interp_errors_np)) if len(interp_errors_np) > 0 else float('inf'),
                    'max_error': float(np.max(interp_errors_np)) if len(interp_errors_np) > 0 else float('inf'),
                    'median_error': float(np.median(interp_errors_np)) if len(interp_errors_np) > 0 else float('inf'),
                    'count': len(interp_errors_np),
                    'percentage': 100 * len(interp_errors_np) / len(valid_errors) if len(valid_errors) > 0 else 0.0
                }
            else:
                metrics['has_interp_data'] = False

            return metrics

        except Exception as e:
            logging.error(f"Error calculating tracking accuracy: {str(e)}")
            logging.error(traceback.format_exc())
            return self._empty_metrics()

    def calculate_inter_player_distances(self, positions_data: Dict[str, pd.DataFrame], ignore_date: bool = True) -> Dict[str, Dict]:
        """
        Calculate distances between players with robust handling of missing data.
        
        Args:
            positions_data: Dictionary mapping player IDs to their position DataFrames
            ignore_date: Whether to ignore the date part of timestamps and match only time-of-day
            
        Returns:
            Dictionary of pair names to dictionaries containing timestamp, distances, 
            and completeness information
        """
        distances = {}
        players = list(positions_data.keys())
        
        # Check if we have at least 2 players with data
        if len(players) < 2:
            logging.warning("Need at least 2 players to calculate distances")
            return {}
            
        # Find global time bounds across all player data
        global_start = min(df['date time'].min() for df in positions_data.values() if not df.empty)
        global_end = max(df['date time'].max() for df in positions_data.values() if not df.empty)
        
        # Log date information for debugging
        logging.info(f"Player data date ranges:")
        for player, df in positions_data.items():
            if not df.empty:
                logging.info(f"  {player}: {df['date time'].min()} to {df['date time'].max()}")
        
        # Convert to time objects if needed for consistency
        if isinstance(global_start, datetime):
            global_start = global_start.time()
        if isinstance(global_end, datetime):
            global_end = global_end.time()
            
        start_seconds = global_start.hour * 3600 + global_start.minute * 60 + global_start.second
        end_seconds = global_end.hour * 3600 + global_end.minute * 60 + global_end.second
        
        # Preprocess data for each player - convert times to seconds
        processed_data = {}
        for player, df in positions_data.items():
            if df.empty:
                logging.warning(f"Empty dataframe for player {player}")
                continue
                
            df_copy = df.copy()
            
            # Handle datetime vs time objects consistently
            if isinstance(df_copy['date time'].iloc[0], datetime):
                if ignore_date:
                    # Extract only time component if ignoring date
                    df_copy['date time'] = df_copy['date time'].dt.time
                # else: keep datetime objects as is
            
            # Convert to seconds since midnight (for time-of-day) or timestamp (for absolute time)
            if ignore_date or isinstance(df_copy['date time'].iloc[0], time):
                # Convert to seconds since midnight, rounded to nearest 0.1s
                df_copy['seconds'] = df_copy['date time'].apply(
                    lambda x: round((x.hour * 3600 + x.minute * 60 + x.second + 
                                (x.microsecond/1e6 if hasattr(x, 'microsecond') else 0)) * 10) / 10
                )
            else:
                # Use actual timestamps
                df_copy['seconds'] = df_copy['date time'].apply(
                    lambda x: round(x.timestamp() * 10) / 10  # Round to nearest 0.1s
                )
                
            processed_data[player] = df_copy

        # Process each pair of players
        for i, player1 in enumerate(players):
            if player1 not in processed_data:
                continue  # Skip players with no data
                
            for j in range(i + 1, len(players)):
                player2 = players[j]
                if player2 not in processed_data:
                    continue  # Skip players with no data
                    
                pair_name = f"{player1}-{player2}"
                
                df1 = processed_data[player1]
                df2 = processed_data[player2]
                
                # Create unified time points covering full protocol period
                time_step = 0.1  # 10Hz - adjust as needed for your data
                time_points = np.arange(start_seconds, end_seconds + time_step, time_step)
                
                distances_list = []
                timestamps = []
                data_quality = []  # Track interpolation quality
                
                # Define tolerance for finding closest points
                tolerance = 0.05  # seconds
                max_interpolation_gap = 0.5  # seconds - maximum gap allowed for interpolation
                
                for t in time_points:
                    # Find closest points within tolerance
                    p1_close = df1[abs(df1['seconds'] - t) <= tolerance]
                    p2_close = df2[abs(df2['seconds'] - t) <= tolerance]
                    
                    if not p1_close.empty and not p2_close.empty:
                        # Both players have data points very close to this time
                        p1 = p1_close.iloc[0]
                        p2 = p2_close.iloc[0]
                        
                        # Calculate Euclidean distance
                        dist = np.sqrt(
                            (p1['x_real_m'] - p2['x_real_m'])**2 +
                            (p1['y_real_m'] - p2['y_real_m'])**2
                        )
                        
                        distances_list.append(dist)
                        timestamps.append(p1['date time'])  # Original timestamp for display
                        data_quality.append('exact')  # Both points are accurate
                        
                    else:
                        # Consider interpolation if appropriate
                        # Only interpolate if the gap is small enough
                        try:
                            # Find points before and after for player 1
                            p1_before = df1[df1['seconds'] <= t]
                            p1_after = df1[df1['seconds'] > t]
                            
                            # Find points before and after for player 2
                            p2_before = df2[df2['seconds'] <= t]
                            p2_after = df2[df2['seconds'] > t]
                            
                            # Check if we have valid points for interpolation
                            if (not p1_before.empty and not p1_after.empty and 
                                not p2_before.empty and not p2_after.empty):
                                
                                # Get the closest points before and after
                                p1_before = p1_before.iloc[-1]  # Last point before t
                                p1_after = p1_after.iloc[0]   # First point after t
                                p2_before = p2_before.iloc[-1]
                                p2_after = p2_after.iloc[0]
                                
                                # Check if the gaps aren't too large for interpolation
                                p1_gap = p1_after['seconds'] - p1_before['seconds']
                                p2_gap = p2_after['seconds'] - p2_before['seconds']
                                
                                if p1_gap <= max_interpolation_gap and p2_gap <= max_interpolation_gap:
                                    # Interpolate positions
                                    p1_frac = (t - p1_before['seconds']) / p1_gap
                                    p2_frac = (t - p2_before['seconds']) / p2_gap
                                    
                                    p1_x = p1_before['x_real_m'] + p1_frac * (p1_after['x_real_m'] - p1_before['x_real_m'])
                                    p1_y = p1_before['y_real_m'] + p1_frac * (p1_after['y_real_m'] - p1_before['y_real_m'])
                                    p2_x = p2_before['x_real_m'] + p2_frac * (p2_after['x_real_m'] - p2_before['x_real_m'])
                                    p2_y = p2_before['y_real_m'] + p2_frac * (p2_after['y_real_m'] - p2_before['y_real_m'])
                                    
                                    # Calculate distance from interpolated positions
                                    dist = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
                                    
                                    # Construct time object for this timestamp
                                    if ignore_date or isinstance(p1_before['date time'], time):
                                        sec_int = int(t)
                                        microsec = int((t - sec_int) * 1000000)
                                        timestamp = time(
                                            hour=sec_int // 3600,
                                            minute=(sec_int % 3600) // 60,
                                            second=sec_int % 60,
                                            microsecond=microsec
                                        )
                                    else:
                                        # Use proper datetime interpolation for full timestamps
                                        time_diff_seconds = (p1_after['date time'] - p1_before['date time']).total_seconds()
                                        if time_diff_seconds > 0:
                                            timestamp = p1_before['date time'] + timedelta(
                                                seconds=p1_frac * time_diff_seconds
                                            )
                                        else:
                                            timestamp = p1_before['date time']
                                    
                                    distances_list.append(dist)
                                    timestamps.append(timestamp)
                                    data_quality.append('interpolated')  # Mark as interpolated
                                else:
                                    # Gap too large for reliable interpolation
                                    continue
                            else:
                                # Not enough points for interpolation
                                continue
                        except Exception as e:
                            logging.debug(f"Interpolation error at t={t}: {str(e)}")
                            continue
                
                if distances_list:
                    distances[pair_name] = {
                        'timestamps': timestamps,
                        'distances': distances_list,
                        'data_quality': data_quality,
                        'time_range': (global_start, global_end),
                        'completeness': len(distances_list) / len(time_points)  # Metric of data completeness
                    }
                    
                    logging.info(f"Calculated {len(distances_list)} distances for {pair_name}")
                    logging.info(f"Distance range: {min(distances_list):.2f}m to {max(distances_list):.2f}m")
                    logging.info(f"Data completeness: {distances[pair_name]['completeness']*100:.1f}%")
                    
                    # Log interpolation stats
                    if 'interpolated' in data_quality:
                        interp_count = data_quality.count('interpolated')
                        interp_pct = interp_count / len(data_quality) * 100
                        logging.info(f"Interpolated points: {interp_count} ({interp_pct:.1f}%)")
                else:
                    logging.warning(f"No common timestamps found for {pair_name}")
            
        return distances
    def _get_interpolated_position(self, df: pd.DataFrame, time_seconds: float,
                            ignore_date: bool = True) -> Optional[Tuple[float, float]]:
        """
        Get interpolated position at given time in seconds.
        
        Args:
            df: DataFrame with position data
            time_seconds: Time in seconds (either seconds-of-day or timestamp)
            ignore_date: Whether time_seconds represents seconds-of-day or timestamp
            
        Returns:
            Tuple of (x, y) position or None if interpolation isn't possible
        """
        try:
            # Find closest points before and after
            before_mask = df['seconds'] <= time_seconds
            after_mask = df['seconds'] > time_seconds
            
            if not any(before_mask) or not any(after_mask):
                return None
                
            before_idx = before_mask.idxmax()
            after_idx = after_mask.idxmin()
            
            if before_idx == after_idx:
                return (df.loc[before_idx, 'x_real_m'], 
                    df.loc[before_idx, 'y_real_m'])
            
            # Linear interpolation
            t1, t2 = df.loc[before_idx, 'seconds'], df.loc[after_idx, 'seconds']
            frac = (time_seconds - t1) / (t2 - t1)
            
            x = df.loc[before_idx, 'x_real_m'] + frac * (df.loc[after_idx, 'x_real_m'] - df.loc[before_idx, 'x_real_m'])
            y = df.loc[before_idx, 'y_real_m'] + frac * (df.loc[after_idx, 'y_real_m'] - df.loc[before_idx, 'y_real_m'])
            
            return (x, y)
            
        except Exception as e:
            logging.error(f"Error interpolating position: {str(e)}")
            return None

    def compare_distance_calculations(self,
                                    ground_truth_distances: Dict[str, pd.DataFrame],
                                    gpexe_distances: Dict[str, pd.DataFrame],
                                    xseed_distances: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Compare accuracy of inter-player distance calculations.
        
        Args:
            ground_truth_distances: Ground truth distances between players
            gpexe_distances: Distances calculated from GPEXE data
            xseed_distances: Distances calculated from Xseed data
            
        Returns:
            Dictionary containing distance comparison metrics
        """
        try:
            comparison_metrics = {}
            
            for pair in ground_truth_distances.keys():
                gt_dist = ground_truth_distances[pair]
                gpexe_dist = gpexe_distances.get(pair, pd.DataFrame())
                xseed_dist = xseed_distances.get(pair, pd.DataFrame())

                # Calculate error metrics for each system
                gpexe_errors = self._calculate_distance_errors(gt_dist, gpexe_dist)
                xseed_errors = self._calculate_distance_errors(gt_dist, xseed_dist)

                comparison_metrics[pair] = {
                    'gpexe': self._calculate_error_metrics(gpexe_errors),
                    'xseed': self._calculate_error_metrics(xseed_errors)
                }

            return comparison_metrics
            
        except Exception as e:
            logging.error(f"Error comparing distance calculations: {str(e)}")
            return {}

    def _calculate_distance_errors(self, gt_distances: pd.DataFrame, 
                                 track_distances: pd.DataFrame) -> np.ndarray:
        """Calculate errors between ground truth and tracked distances."""
        if gt_distances.empty or track_distances.empty:
            return np.array([])

        # Merge on time and calculate differences
        merged = pd.merge_asof(
            gt_distances.sort_values('time'),
            track_distances.sort_values('time'),
            on='time',
            tolerance=pd.Timedelta('1s')
        )

        if not merged.empty:
            return np.abs(merged['distance_x'] - merged['distance_y'])
        return np.array([])

    def _calculate_error_metrics(self, errors: np.ndarray) -> Dict[str, float]:
        """Calculate standard error metrics from array of errors."""
        if len(errors) == 0:
            return {
                'mean_error': np.nan,
                'rmse': np.nan,
                'max_error': np.nan,
                'matched_points': 0
            }

        return {
            'mean_error': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(np.square(errors)))),
            'max_error': float(np.max(errors)),
            'matched_points': len(errors)
        }

    def analyze_tracking_performance(self, ground_truth: pd.DataFrame, tracking_data: pd.DataFrame, 
                              time_start: time, time_end: time) -> Dict:
        """
        Analyze tracking performance by comparing ground truth with tracking data.
        
        Args:
            ground_truth (pd.DataFrame): Ground truth data with columns ['date time', 'x_real_m', 'y_real_m']
            tracking_data (pd.DataFrame): Tracking data with columns ['date time', 'x_real_m', 'y_real_m']
            time_start (time): Start time of analysis window
            time_end (time): End time of analysis window
        
        Returns:
            Dict: Performance metrics
        """
        try:
            # Log data samples for debugging
            logging.info("\nGround Truth Sample:")
            logging.info(ground_truth.head())
            logging.info("\nTracking Data Sample:")
            logging.info(tracking_data.head())

            # Convert time objects to datetime if necessary
            base_date = datetime.now().date()
            
            if isinstance(ground_truth['date time'].iloc[0], time):
                ground_truth = ground_truth.copy()
                ground_truth['date time'] = ground_truth['date time'].apply(
                    lambda x: datetime.combine(base_date, x)
                )
            
            if isinstance(tracking_data['date time'].iloc[0], time):
                tracking_data = tracking_data.copy()
                tracking_data['date time'] = tracking_data['date time'].apply(
                    lambda x: datetime.combine(base_date, x)
                )

            # Calculate total time period
            total_time = (ground_truth['date time'].max() - 
                        ground_truth['date time'].min()).total_seconds()

            # Initialize error arrays
            position_errors = []
            x_errors = []
            y_errors = []
            
            # Set threshold for significant errors (in meters)
            error_threshold = 5.0  # meters

            # Process each ground truth point
            for idx, gt_point in ground_truth.iterrows():
                # Find closest tracking point in time
                time_diff = abs(tracking_data['date time'] - gt_point['date time'])
                closest_idx = time_diff.idxmin()
                track_point = tracking_data.loc[closest_idx]
                
                # Calculate position error
                x_error = abs(gt_point['x_real_m'] - track_point['x_real_m'])
                y_error = abs(gt_point['y_real_m'] - track_point['y_real_m'])
                position_error = np.sqrt(x_error**2 + y_error**2)
                
                # Store errors
                x_errors.append(x_error)
                y_errors.append(y_error)
                position_errors.append(position_error)
                
                # Log significant errors
                if position_error > error_threshold:
                    logging.info(f"Point {idx} exceeded threshold: error={position_error:.2f}m")

            # Calculate error statistics
            mean_position_error = np.mean(position_errors)
            max_position_error = np.max(position_errors)
            std_position_error = np.std(position_errors)
            
            mean_x_error = np.mean(x_errors)
            mean_y_error = np.mean(y_errors)
            
            # Calculate percentage of points exceeding threshold
            points_above_threshold = sum(1 for error in position_errors if error > error_threshold)
            error_percentage = (points_above_threshold / len(position_errors)) * 100

            # Calculate tracking continuity
            tracking_intervals = np.diff(tracking_data['date time'].values)
            max_gap = np.max(tracking_intervals).total_seconds()
            mean_gap = np.mean(tracking_intervals).total_seconds()

            # Compile metrics
            metrics = {
                'total_time': total_time,
                'mean_position_error': mean_position_error,
                'max_position_error': max_position_error,
                'std_position_error': std_position_error,
                'mean_x_error': mean_x_error,
                'mean_y_error': mean_y_error,
                'error_percentage': error_percentage,
                'max_gap': max_gap,
                'mean_gap': mean_gap,
                'points_analyzed': len(position_errors),
                'points_above_threshold': points_above_threshold
            }

            return metrics

        except Exception as e:
            logging.error(f"Error analyzing tracking performance: {str(e)}")
            if 'metrics' not in locals():
                metrics = {
                    'total_time': 0,
                    'mean_position_error': 0,
                    'max_position_error': 0,
                    'std_position_error': 0,
                    'mean_x_error': 0,
                    'mean_y_error': 0,
                    'error_percentage': 0,
                    'max_gap': 0,
                    'mean_gap': 0,
                    'points_analyzed': 0,
                    'points_above_threshold': 0
                }
            return metrics

    def calculate_detailed_metrics(self, ground_truth: pd.DataFrame, 
                                 tracking_data: pd.DataFrame,
                                 time_tolerance: int) -> Dict:
        """Calculate detailed metrics for tracking system."""
        if ground_truth.empty or tracking_data.empty:
            return self._empty_detailed_metrics()

        try:
            # Basic accuracy metrics
            accuracy_metrics = self.calculate_tracking_accuracy(
                ground_truth, tracking_data, time_tolerance
            )

            # Velocity metrics if available
            velocity_metrics = self._calculate_velocity_metrics(tracking_data)

            # Combine all metrics
            return {
                **accuracy_metrics,
                'velocity': velocity_metrics
            }

        except Exception as e:
            logging.error(f"Error calculating detailed metrics: {str(e)}")
            return self._empty_detailed_metrics()

    def _calculate_velocity_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate velocity-related metrics."""
        try:
            if 'speed' not in data.columns:
                return {
                    'mean_speed': 0.0,
                    'max_speed': 0.0,
                    'std_speed': 0.0
                }

            return {
                'mean_speed': float(data['speed'].mean()),
                'max_speed': float(data['speed'].max()),
                'std_speed': float(data['speed'].std()),
                'total_distance': float(np.sum(data['speed'] * 0.1))  # Assuming 10Hz sampling
            }
        except Exception as e:
            logging.error(f"Error calculating velocity metrics: {str(e)}")
            return {
                'mean_speed': 0.0,
                'max_speed': 0.0,
                'std_speed': 0.0,
                'total_distance': 0.0
            }

    def _calculate_system_differences(self, gpexe_metrics: Dict, 
                                    xseed_metrics: Dict) -> Dict:
        """Calculate differences between tracking systems."""
        return {
            'mean_error_diff': gpexe_metrics['mean_error'] - xseed_metrics['mean_error'],
            'rmse_diff': gpexe_metrics['rmse'] - xseed_metrics['rmse'],
            'matching_rate_diff': gpexe_metrics['matching_rate'] - xseed_metrics['matching_rate'],
            'velocity_diff': {
                'mean_speed': gpexe_metrics['velocity']['mean_speed'] - xseed_metrics['velocity']['mean_speed'],
                'max_speed': gpexe_metrics['velocity']['max_speed'] - xseed_metrics['velocity']['max_speed']
            }
        }

    def _empty_metrics(self) -> Dict:
        """Return empty basic metrics dictionary."""
        return {
            'mean_error': 0.0,
            'rmse': 0.0,
            'max_error': 0.0,
            'matching_rate': 0.0,
            'error_distribution': []
        }

    def _empty_detailed_metrics(self) -> Dict:
        """Return empty detailed metrics dictionary."""
        return {
            **self._empty_metrics(),
            'velocity': {
                'mean_speed': 0.0,
                'max_speed': 0.0,
                'std_speed': 0.0,
                'total_distance': 0.0
            }
        }
    def convert_to_seconds(self, timestamps, ignore_date=True):
        """
        Convert timestamps to seconds representation based on ignore_date setting.
        
        Args:
            timestamps: List or Series of datetime or time objects
            ignore_date: Whether to extract only seconds-of-day (True) or use full timestamps (False)
            
        Returns:
            Numpy array of seconds values
        """
        if ignore_date:
            # Extract seconds-of-day (time-of-day)
            return np.array([
                ts.hour * 3600 + ts.minute * 60 + ts.second + 
                (ts.microsecond/1e6 if hasattr(ts, 'microsecond') else 0)
                for ts in timestamps
            ])
        else:
            # Use actual timestamps
            return np.array([
                ts.timestamp() if isinstance(ts, datetime) else 
                datetime.combine(datetime.now().date(), ts).timestamp()
                for ts in timestamps
            ])
    
    def analyze_distances_over_time(self, positions_data: Dict[str, pd.DataFrame], 
                          system_type: str, ignore_date: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Analyze inter-player distances over time for a specific tracking system.
        Returns both raw distances and time-synchronized comparison with ground truth.
        
        Args:
            positions_data: Dictionary mapping player IDs to position DataFrames  
            system_type: String identifier for the tracking system
            ignore_date: Whether to ignore dates and compare only time-of-day
            
        Returns:
            Dictionary mapping player pairs to distance time series DataFrames
        """
        time_series_data = {}
        
        # Preprocess data to use consistent time representation
        processed_data = {}
        for player, df in positions_data.items():
            if df.empty:
                continue
                
            df_copy = df.copy()
            
            # Convert timestamps to seconds representation based on ignore_date
            if 'seconds' not in df_copy.columns:
                if ignore_date:
                    # Extract time-of-day as seconds since midnight
                    if isinstance(df_copy['date time'].iloc[0], datetime):
                        df_copy['seconds'] = df_copy['date time'].apply(
                            lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond/1e6
                        )
                    else:  # Already time objects
                        df_copy['seconds'] = df_copy['date time'].apply(
                            lambda x: x.hour * 3600 + x.minute * 60 + x.second + 
                                    (x.microsecond/1e6 if hasattr(x, 'microsecond') else 0)
                        )
                else:
                    # Use actual timestamps
                    df_copy['seconds'] = df_copy['date time'].apply(
                        lambda x: x.timestamp() if isinstance(x, datetime) else 
                            datetime.combine(datetime.now().date(), x).timestamp()
                    )
                    
            processed_data[player] = df_copy
        
        # Find common time range across all players
        common_start = max(df['seconds'].min() for df in processed_data.values())
        common_end = min(df['seconds'].max() for df in processed_data.values())
        time_points = np.arange(common_start, common_end, 0.1)  # 10Hz sampling
        
        # Get player pairs
        players = list(processed_data.keys())
        
        for i, player1 in enumerate(players):
            for j in range(i + 1, len(players)):
                player2 = players[j]
                
                # Create a DataFrame to store time-series data
                distances_df = pd.DataFrame({
                    'seconds': time_points
                })
                
                # Calculate distances for each time point
                distances = []
                timestamps = []
                
                for t in time_points:
                    pos1 = self._get_interpolated_position(processed_data[player1], t, ignore_date)
                    pos2 = self._get_interpolated_position(processed_data[player2], t, ignore_date)
                    
                    if pos1 is not None and pos2 is not None:
                        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        distances.append(dist)
                        
                        # Convert seconds back to timestamp for display
                        if ignore_date:
                            # Create time object
                            seconds_int = int(t)
                            microseconds = int((t - seconds_int) * 1000000)
                            ts = datetime.combine(
                                datetime.now().date(),
                                time(
                                    hour=seconds_int // 3600,
                                    minute=(seconds_int % 3600) // 60,
                                    second=seconds_int % 60,
                                    microsecond=microseconds
                                )
                            )
                        else:
                            # Create full datetime from timestamp
                            ts = datetime.fromtimestamp(t)
                        
                        timestamps.append(ts)
                    else:
                        distances.append(np.nan)
                        timestamps.append(None)
                
                # Filter out None timestamps
                valid_data = [(ts, dist) for ts, dist in zip(timestamps, distances) if ts is not None]
                if valid_data:
                    valid_timestamps, valid_distances = zip(*valid_data)
                    
                    distances_df = pd.DataFrame({
                        'time': valid_timestamps,
                        'seconds': [t for i, t in enumerate(time_points) if timestamps[i] is not None],
                        f'distance_{system_type}': valid_distances
                    })
                    
                    pair_name = f"{player1}-{player2}"
                    time_series_data[pair_name] = distances_df
                else:
                    logging.warning(f"No valid distance calculations for pair {player1}-{player2}")
            
        return time_series_data

    def compare_distances_over_time(self, ground_truth_data: Dict[str, pd.DataFrame],
                              tracking_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compare tracking system distances with ground truth over time.
        Returns error analysis and creates visualizations.
        """
        error_analysis = {}
        
        for pair in ground_truth_data.keys():
            gt_df = ground_truth_data[pair]
            track_df = tracking_data[pair]
            
            # Merge ground truth and tracking data
            merged_df = pd.merge_asof(
                track_df.sort_values('seconds'),
                gt_df.sort_values('seconds'),
                on='seconds',
                tolerance=0.1  # 100ms tolerance for matching
            )
            
            # Calculate absolute errors
            merged_df['distance_error'] = abs(
                merged_df['distance_tracking'] - merged_df['distance_ground_truth']
            )
            
            error_analysis[pair] = merged_df
            
        return error_analysis 