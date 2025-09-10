import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta, time
from src.ground_truth_processor import GroundTruthProcessor
import os
from src.data_sync import DataSynchronizer
from services.haversine_service import HaversineService
from services.pitch_service import PitchService
import logging
import traceback
from pathlib import Path

# Standard column mappings for GPS data
COLUMN_MAPPING = {
    'lat': 'latitude',
    'lng': 'longitude'
}

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names in dataframe."""
    return df.rename(columns=COLUMN_MAPPING)


class DataProcessor:
    def __init__(self, session_id: str):
        """Initialize DataProcessor with data sync capabilities."""
        self.session_id = session_id
        stadium_path = f'data/{session_id}/stadium_properties.csv'
        
        self.pitch_service = PitchService(stadium_path)
        self.pitch_result = self.pitch_service.get_pitch()
        self.corner1_lat = self.pitch_result['origin']['lat']
        self.corner1_lon = self.pitch_result['origin']['lng']
        self.corner2_lat = self.pitch_result['second_point']['lat']
        self.corner2_lon = self.pitch_result['second_point']['lng']
        
        self.base_path = Path('data') / session_id

        # Detect available players in this session
        self.players = self._detect_available_players()
        logging.info(f"Detected {len(self.players)} players in session {session_id}: {', '.join(self.players)}")
        
        # Define file paths for available players 
        self.input_files = {}
        self.activity_files = {}
        self.protocol_files = {}
        self.shinguard_files = {}
        
        # For each player, handle different hardware types
        for player in self.players:
            player_num = player.split()[-1]  # Extract player number
            
            # GPExe uses base player number (common for both hardware types)
            self.input_files[player] = self._find_file(f'gpexe_track_{player_num}')
            
            # For protocol and activity files, handle both hardware types
            self.protocol_files[player] = {}
            self.activity_files[player] = {}
            
            # Old hardware (base player number)
            self.protocol_files[player]['old'] = self._find_file(f'Protocol_Sheet_{player_num}')
            self.activity_files[player]['old'] = self._find_file(f'Activity_Sheet_{player_num}')
            
            # New hardware (player number with .5)
            decimal_num = f"{player_num}.5"
            self.protocol_files[player]['new'] = self._find_file(f'Protocol_Sheet_{decimal_num}')
            self.activity_files[player]['new'] = self._find_file(f'Activity_Sheet_{decimal_num}')
            
            # Setup shinguard file paths for both hardware types
            player_dir = f'player_{player_num.replace(".", "_")}'
            self.shinguard_files[player] = {}
            
            # Old hardware files
            old_file_found = False
            for filename in ['df_trace_old.csv', 'df_trace_old.xlsx', 'df_trace_old.xls']:
                old_file = self.base_path / f'{player_dir}/{filename}'
                if old_file.exists():
                    self.shinguard_files[player]['old'] = old_file
                    old_file_found = True
                    break
            if not old_file_found:
                # Try traditional files as fallback for old hardware
                for filename in ['df_trace.csv', 'df_trace.xlsx', 'df_trace.xls']:
                    fallback = self.base_path / f'{player_dir}/{filename}'
                    if fallback.exists():
                        self.shinguard_files[player]['old'] = fallback
                        old_file_found = True
                        break
                if not old_file_found:
                    self.shinguard_files[player]['old'] = self.base_path / f'{player_dir}/df_trace_old.csv'

            # New hardware files
            new_file_found = False
            for filename in ['df_trace_new.csv', 'df_trace_new.xlsx', 'df_trace_new.xls']:
                new_file = self.base_path / f'{player_dir}/{filename}'
                if new_file.exists():
                    self.shinguard_files[player]['new'] = new_file
                    new_file_found = True
                    break
            if not new_file_found:
                # Try decimal subdirectory as fallback for new hardware
                decimal_dir = f'player_{decimal_num.replace(".", "_")}'
                for filename in ['df_trace.csv', 'df_trace.xlsx', 'df_trace.xls']:
                    fallback = self.base_path / f'{decimal_dir}/{filename}'
                    if fallback.exists():
                        self.shinguard_files[player]['new'] = fallback
                        new_file_found = True
                        break
                if not new_file_found:
                    self.shinguard_files[player]['new'] = self.base_path / f'{player_dir}/df_trace_new.csv'

        # Detect which hardware types are available in this session
        self.hardware_types = self._detect_session_hardware()
        logging.info(f"Hardware types detected: Old={self.hardware_types['old']}, New={self.hardware_types['new']}")

        # Initialize ground truth processor with first player data
        # We'll use old hardware by default for initialization
        first_player = self.players[0] if self.players else 'Player 1'
        default_hw = 'old' if self.hardware_types['old'] else 'new'
        
        self.ground_truth_processor = GroundTruthProcessor(
            str(self.protocol_files.get(first_player, {}).get(default_hw, '')),
            [str(self.activity_files.get(first_player, {}).get(default_hw, ''))],
            self.pitch_service,
            self.session_id
        )

        self.haversine = HaversineService()
        
        # Initialize the data sync component
        from src.data_sync import DataSynchronizer
        self.data_sync = DataSynchronizer()
        
        self._verify_required_files(skip_missing=True)


    def _detect_session_hardware(self) -> Dict[str, bool]:
        """Detect which hardware types are available in this session."""
        has_old = False
        has_new = False
        
        for player in self.players:
            # Check for old hardware files
            old_hardware_indicators = [
                self._check_file_exists(f'Protocol_Sheet_{player[-1]}'),
                self._check_file_exists(f'player_{player[-1].replace(".", "_")}/df_trace_old'),
                self._check_file_exists(f'player_{player[-1].replace(".", "_")}/df_trace')
            ]
            if any(old_hardware_indicators):
                has_old = True
                
            # Check for new hardware files
            decimal_num = f"{player[-1]}.5"
            new_hardware_indicators = [
                self._check_file_exists(f'Protocol_Sheet_{decimal_num}'),
                self._check_file_exists(f'player_{player[-1].replace(".", "_")}/df_trace_new'),
                self._check_file_exists(f'player_{decimal_num.replace(".", "_")}/df_trace')
            ]
            if any(new_hardware_indicators):
                has_new = True
                
            if has_old and has_new:
                break
        
        return {'old': has_old, 'new': has_new}



    def load_file(file_path: Union[str, Path], default_sep=";") -> pd.DataFrame:
        """
        Universal file loader that handles CSV, XLS, and XLSX formats.
        
        Args:
            file_path: Path to the file
            default_sep: Default separator for CSV files
            
        Returns:
            Loaded DataFrame or empty DataFrame if file doesn't exist
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path.exists():
            logging.warning(f"File not found: {path}")
            return pd.DataFrame()
        
        try:
            suffix = path.suffix.lower()
            if suffix in ['.xlsx', '.xls']:
                # Excel file
                return pd.read_excel(path)
            elif suffix == '.csv':
                # CSV file - try with specified separator first
                try:
                    return pd.read_csv(path, sep=default_sep)
                except:
                    # If that fails, try with auto-detection
                    return pd.read_csv(path, sep=None, engine='python')
            else:
                # Unknown format - try CSV first, then Excel
                try:
                    return pd.read_csv(path, sep=default_sep)
                except:
                    try:
                        return pd.read_excel(path)
                    except:
                        logging.error(f"Could not determine format for file: {path}")
                        return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading file {path}: {str(e)}")
            return pd.DataFrame()
    def transform_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform GPS coordinates to local pitch coordinate system.
        
        Args:
            df: DataFrame containing latitude/longitude coordinates
        """
        try:
            logging.info("Starting coordinate transformation")
            logging.info(f"Input columns: {df.columns.tolist()}")
            
            # First ensure we have the required columns
            if 'lat' in df.columns and 'lng' in df.columns:
                lat_col, lng_col = 'lat', 'lng'
            elif 'latitude' in df.columns and 'longitude' in df.columns:
                lat_col, lng_col = 'latitude', 'longitude'
            else:
                logging.error("No latitude/longitude columns found")
                logging.info(f"Available columns: {df.columns.tolist()}")
                raise ValueError("Missing coordinate columns")

            transformed_data = []
            origin_point = {"lat": self.corner1_lat, "lng": self.corner1_lon}
            
            for _, row in df.iterrows():
                if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
                    continue
                    
                target_point = {"lat": row[lat_col], "lng": row[lng_col]}
                # Calculate distance from origin point to get local coordinates
                result = HaversineService.distance_haversine(origin_point, target_point, meters=1000, rotation=self.pitch_service.rotation)
                
                if result["dx"] is not None and result["dy"] is not None:
                    transformed_data.append({
                        'date time': row['date time'],
                        'x_real_m': result["dx"],
                        'y_real_m': result["dy"],
                        'latitude': row[lat_col],
                        'longitude': row[lng_col],
                        'speed': row['speed (km/h)'] if 'speed (km/h)' in df.columns else None
                    })
        
            transformed_df = pd.DataFrame(transformed_data)
            
            if not transformed_df.empty:
                logging.info(f"Transformed {len(transformed_df)} points")
                logging.info("Sample of transformed coordinates:")
                logging.info(transformed_df[['x_real_m', 'y_real_m']].head())
            else:
                logging.warning("No points were successfully transformed")
            
            return transformed_df

        except Exception as e:
            logging.error(f"Transform coordinates error: {str(e)}")
            logging.error(f"Input DataFrame columns: {df.columns.tolist()}")
            raise

    def transform_to_local_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert single latitude/longitude point to local coordinates.
        Used for individual point transformations.
        """
        origin_point = {"lat": self.corner1_lat, "lng": self.corner1_lon}
        target_point = {"lat": lat, "lng": lon}
        
        result = HaversineService.distance_haversine(origin_point, target_point, meters=1000, rotation=self.pitch_service.rotation)
        return result['dx'], result['dy']

    def _verify_required_files(self, skip_missing=False):
        """Verify required files exist, with option to skip missing files."""
        missing_files = []
        
        # Check essential files (always required)
        essential_files = [
            self.base_path / 'stadium_properties.csv'
        ]
        
        for file_path in essential_files:
            if not file_path.exists():
                # Try other extensions
                found = False
                for ext in ['.xlsx', '.xls']:
                    alt_path = file_path.with_suffix(ext)
                    if alt_path.exists():
                        found = True
                        break
                if not found:
                    missing_files.append(str(file_path))

        # Check other files (can be skipped with warning)
        # Modified to handle nested dictionaries for hardware types
        missing_optional = []
        
        # Check input files (these are still Path objects)
        for player, file_path in self.input_files.items():
            if not file_path.exists():
                missing_optional.append(f"Input file for {player}: {file_path}")
        
        # Check activity and protocol files (these are now dicts with 'old' and 'new' keys)
        for player, hw_files in self.activity_files.items():
            for hw_type, file_path in hw_files.items():
                if not file_path.exists():
                    missing_optional.append(f"Activity file for {player} ({hw_type} hardware): {file_path}")
        
        for player, hw_files in self.protocol_files.items():
            for hw_type, file_path in hw_files.items():
                if not file_path.exists():
                    missing_optional.append(f"Protocol file for {player} ({hw_type} hardware): {file_path}")
        
        # Check shinguard files
        for player, hw_files in self.shinguard_files.items():
            # For shinguard files, only need one hardware type to exist per player
            if not any(file_path.exists() for hw_type, file_path in hw_files.items()):
                missing_optional.append(f"No shinguard files found for {player}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing essential files in {self.session_id}:\n" + 
                                "\n".join(missing_files))
        
        if missing_optional and not skip_missing:
            raise FileNotFoundError(f"Missing optional files in {self.session_id}:\n" + 
                                "\n".join(missing_optional))
        elif missing_optional:
            logging.warning(f"Missing optional files in {self.session_id}:\n" + 
                        "\n".join(missing_optional))

    def get_players_list(self) -> List[str]:
        """Return list of available players."""
        return self.players

    def get_protocol_info(self, player: str, hardware_type: str = 'old') -> pd.DataFrame:
        """Get protocol info for a specific hardware type."""
        player_num = player.split()[-1]
        
        if hardware_type == 'old':
            protocol_file = self._find_file(f'Protocol_Sheet_{player_num}')
            activity_file = self._find_file(f'Activity_Sheet_{player_num}')
        else:  # new hardware
            decimal_num = f"{player_num}.5"
            protocol_file = self._find_file(f'Protocol_Sheet_{decimal_num}')
            activity_file = self._find_file(f'Activity_Sheet_{decimal_num}')
        
        if not protocol_file.exists() or not activity_file.exists():
            logging.warning(f"Protocol or activity file not found for {player} with {hardware_type} hardware")
            return pd.DataFrame()
        
        gt_processor = GroundTruthProcessor(
            str(protocol_file),
            [str(activity_file)],
            self.pitch_service,
            self.session_id
        )
        return gt_processor.protocols_df

    def load_pitch_data(self) -> Optional[pd.DataFrame]:
        """
        Load and process pitch outline data.
        
        Returns:
            DataFrame with pitch corner coordinates in meters
        """
        try:
            # Create corner points in local coordinate system
            corners = [
                {'x_real_m': 0, 'y_real_m': 0},  # Origin point
                {'x_real_m': self.pitch_result['pitch_width'], 'y_real_m': 0},  # Width point
                {'x_real_m': self.pitch_result['pitch_width'], 
                 'y_real_m': self.pitch_result['pitch_height']},  # Far corner
                {'x_real_m': 0, 'y_real_m': self.pitch_result['pitch_height']},  # Height point
            ]
            
            return pd.DataFrame(corners)
            
        except Exception as e:
            logging.error(f"Error loading pitch data: {str(e)}")
            return None
    def process_player_data(self, player: str, protocol_id: int, time_tolerance: int, 
                   hardware_type: str = 'old') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Process player data for a specific hardware type with consistent sampling."""
        try:
            protocol_file = self.protocol_files[player][hardware_type]
            activity_file = self.activity_files[player][hardware_type]
            
            if not protocol_file.exists() or not activity_file.exists():
                logging.warning(f"Protocol or activity file not found for {player} with {hardware_type} hardware")
                return pd.DataFrame(), pd.DataFrame(), {}
            
            gt_processor = GroundTruthProcessor(
                str(protocol_file),
                [str(activity_file)],
                self.pitch_service,
                self.session_id
            )

            protocol_df = gt_processor.protocols_df
            if protocol_df.empty:
                logging.warning(f"No protocols found for {player} with {hardware_type} hardware")
                return pd.DataFrame(), pd.DataFrame(), {}
                    
            protocol_rows = protocol_df[protocol_df['Protocol ID'] == protocol_id]
            if protocol_rows.empty:
                logging.warning(f"Protocol ID {protocol_id} not found for {player} with {hardware_type} hardware")
                return pd.DataFrame(), pd.DataFrame(), {}
                    
            protocol_info = protocol_rows.iloc[0].to_dict()  # Convert to dict to handle missing columns
            
            # Ensure protocol_info has required keys
            if 'Start_Time' not in protocol_info or 'End_Time' not in protocol_info:
                logging.warning(f"Missing Start_Time or End_Time in protocol for {player}")
                # Add default values if missing
                if 'Start_Time' not in protocol_info:
                    protocol_info['Start_Time'] = datetime.now().time()
                if 'End_Time' not in protocol_info:
                    protocol_info['End_Time'] = (datetime.now() + timedelta(minutes=1)).time()
            
            logging.info(f"Processing protocol {protocol_id} for {player} with {hardware_type} hardware")
            logging.info(f"Protocol time range: {protocol_info['Start_Time']} to {protocol_info['End_Time']}")
            
            # Load and process ground truth data
            ground_truth = gt_processor.get_ground_truth_path(protocol_id)
            if not ground_truth.empty:
                logging.info(f"Generated {len(ground_truth)} ground truth points")
                logging.info("Sample ground truth coordinates:")
                logging.info(ground_truth[['x_real_m', 'y_real_m', 'is_circular']].head())
                
                # Calculate original sampling rate
                if len(ground_truth) > 1:
                    gt_time_range = (ground_truth['date time'].max() - ground_truth['date time'].min()).total_seconds()
                    gt_sampling_rate = len(ground_truth) / gt_time_range if gt_time_range > 0 else 0
                    logging.info(f"Ground truth original sampling rate: {gt_sampling_rate:.2f} Hz")
                
                # Resample ground truth to 10Hz
                ground_truth = self.data_sync.resample_data(ground_truth, target_frequency='100ms')
                logging.info(f"Resampled ground truth to 10Hz: {len(ground_truth)} points")
                
                # Add hardware type to ground truth data
                ground_truth['hardware_type'] = hardware_type
            
            # Load GPS data with debug logging
            gps_df = pd.DataFrame()  # Initialize as empty DataFrame
            try:
                if player in self.input_files and self.input_files[player].exists():
                    logging.info(f"Loading GPExe data from: {self.input_files[player]}")
                    file_path = self.input_files[player]
                    
                    # Try multiple loading approaches based on file extension
                    if str(file_path).endswith(('.xlsx', '.xls')):
                        gps_df = pd.read_excel(file_path)
                    else:
                        try:
                            gps_df = pd.read_csv(file_path, sep=";")
                        except:
                            gps_df = pd.read_csv(file_path, sep=None, engine='python')
                else:
                    logging.warning(f"No GPExe file found for {player}. This is expected for some players.")
                    # Return the ground truth with empty GPS data but valid protocol info
                    return ground_truth, pd.DataFrame(), protocol_info
                    
            except Exception as file_error:
                logging.error(f"Error loading GPExe data: {str(file_error)}")
                # Still return the ground truth with valid protocol info
                return ground_truth, pd.DataFrame(), protocol_info
                    
            # Process GPS data only if we have any
            if not gps_df.empty:
                logging.info(f"Raw GPExe data columns: {gps_df.columns.tolist()}")
                logging.info(f"Raw GPExe data sample:")
                logging.info(gps_df.head())
                
                # Process date/time values
                try:
                    # First check if 'date time' column exists, if not, try to find a suitable column
                    if 'date time' not in gps_df.columns:
                        datetime_cols = [col for col in gps_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                        if datetime_cols:
                            # Use the first time-related column found
                            gps_df['date time'] = gps_df[datetime_cols[0]]
                            logging.info(f"Using {datetime_cols[0]} as datetime column")
                        else:
                            raise ValueError("No datetime column found in GPEXE data")
                    
                    # Try to convert to datetime with flexible parsing
                    gps_df['date time'] = pd.to_datetime(gps_df['date time'], errors='coerce')
                    
                    # Check for NaT values
                    if gps_df['date time'].isna().any():
                        logging.warning(f"Some datetime values could not be parsed: {gps_df['date time'].isna().sum()} NaT values")
                        gps_df = gps_df.dropna(subset=['date time'])
                        logging.info(f"After dropping NaT values: {len(gps_df)} rows")
                except Exception as dt_error:
                    logging.error(f"Error processing datetime: {str(dt_error)}")
                    return ground_truth, pd.DataFrame(), protocol_info
                        
                # Get protocol times
                protocol_start_time = protocol_info['Start_Time']
                protocol_end_time = protocol_info['End_Time']
                
                # Ensure protocol times are time objects
                if not isinstance(protocol_start_time, time):
                    try:
                        protocol_start_time = pd.to_datetime(protocol_start_time).time()
                    except:
                        protocol_start_time = time(0, 0, 0)  # Default to midnight
                        
                if not isinstance(protocol_end_time, time):
                    try:
                        protocol_end_time = pd.to_datetime(protocol_end_time).time()
                    except:
                        protocol_end_time = time(23, 59, 59)  # Default to end of day
                
                # Get the date from GPS data
                protocol_date = gps_df['date time'].dt.date.iloc[0] if not gps_df.empty else datetime.now().date()
                
                # Create full datetime objects for start and end
                start_datetime = datetime.combine(protocol_date, protocol_start_time)
                end_datetime = datetime.combine(protocol_date, protocol_end_time)
                
                # Apply time tolerance
                if time_tolerance < 0:
                    start_datetime += timedelta(seconds=time_tolerance)
                elif time_tolerance > 0:
                    end_datetime += timedelta(seconds=time_tolerance)
                
                # Filter by datetime
                time_mask = (gps_df['date time'] >= start_datetime) & (gps_df['date time'] <= end_datetime)
                gps_df = gps_df[time_mask].copy()
                
                logging.info(f"Filtered GPExe data: {len(gps_df)} points with valid timestamps")
                if not gps_df.empty:
                    logging.info(f"Sample of filtered data:")
                    logging.info(gps_df[['date time']].head())
                    
                    # Check for coordinate columns
                    coord_cols = {
                        'lat': ['lat', 'latitude', 'gps_lat'],
                        'lng': ['lng', 'long', 'longitude', 'gps_lng']
                    }
                    
                    # Find actual column names for coordinates
                    lat_col = None
                    lng_col = None
                    
                    for col in gps_df.columns:
                        col_lower = col.lower()
                        if any(variant in col_lower for variant in coord_cols['lat']):
                            lat_col = col
                        if any(variant in col_lower for variant in coord_cols['lng']):
                            lng_col = col
                    
                    if not lat_col or not lng_col:
                        logging.error(f"Could not find coordinate columns in GPEXE data")
                        logging.info(f"Available columns: {gps_df.columns.tolist()}")
                        return ground_truth, pd.DataFrame(), protocol_info
                        
                    # Rename coordinate columns to standard format
                    gps_df = gps_df.rename(columns={lat_col: 'latitude', lng_col: 'longitude'})
                    
                    # Drop any rows with NaN coordinates
                    gps_df = gps_df.dropna(subset=['latitude', 'longitude'])
                    logging.info(f"Points with valid coordinates: {len(gps_df)}")
                    
                    # Transform coordinates
                    if not gps_df.empty:
                        logging.info("GPExe data before transformation:")
                        logging.info(gps_df[['latitude', 'longitude']].head())
                        
                        # Transform coordinates to pitch coordinate system
                        transformed_data = []
                        origin_point = {"lat": self.corner1_lat, "lng": self.corner1_lon}
                        
                        for _, row in gps_df.iterrows():
                            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                                continue
                                
                            target_point = {"lat": row['latitude'], "lng": row['longitude']}
                            result = HaversineService.distance_haversine(origin_point, target_point, meters=1000,rotation=self.pitch_service.rotation)
                            
                            if result["dx"] is not None and result["dy"] is not None:
                                new_row = {
                                    'date time': row['date time'],
                                    'x_real_m': result["dx"],
                                    'y_real_m': result["dy"],
                                    'latitude': row['latitude'],
                                    'longitude': row['longitude']
                                }
                                
                                # Add speed if available
                                speed_cols = ['speed', 'speed (km/h)', 'gps_speed']
                                for speed_col in speed_cols:
                                    if speed_col in row:
                                        new_row['speed'] = row[speed_col]
                                        break
                                
                                transformed_data.append(new_row)
                        
                        if transformed_data:
                            gps_df = pd.DataFrame(transformed_data)
                            logging.info("GPExe data after transformation:")
                            logging.info(gps_df[['x_real_m', 'y_real_m']].head())
                            
                            # Calculate original sampling rate
                            if len(gps_df) > 1:
                                gpexe_time_range = (gps_df['date time'].max() - gps_df['date time'].min()).total_seconds()
                                gpexe_sampling_rate = len(gps_df) / gpexe_time_range if gpexe_time_range > 0 else 0
                                logging.info(f"GPEXE original sampling rate: {gpexe_sampling_rate:.2f} Hz")
                            
                            # Resample GPEXE data to 10Hz
                            gps_df = self.data_sync.downsample_gpexe(gps_df)
                            logging.info(f"Resampled GPEXE data to 10Hz: {len(gps_df)} points")
                            
                            # Add hardware type to GPS data
                            gps_df['hardware_type'] = hardware_type
                            
                            # Handle speed data
                            try:
                                if 'speed (km/h)' in gps_df.columns:
                                    gps_df['gps_speed_mean'] = gps_df['speed (km/h)']
                                    logging.info("Speed data processed successfully")
                                else:
                                    logging.warning("Speed data not available in GPS data")
                            except Exception as speed_error:
                                logging.error(f"Error processing speed data: {str(speed_error)}")
                        else:
                            logging.warning("No points remained after coordinate transformation")
                            return ground_truth, pd.DataFrame(), protocol_info
                    else:
                        logging.warning("No valid GPS points found")
                        return ground_truth, pd.DataFrame(), protocol_info
                else:
                    logging.warning("No GPEXE data found in time window")
                    return ground_truth, pd.DataFrame(), protocol_info
            
            # Always ensure protocol_info has required fields
            protocol_info['hardware_type'] = hardware_type
            if 'Start_Time' not in protocol_info:
                protocol_info['Start_Time'] = datetime.now().time()
            if 'End_Time' not in protocol_info:
                protocol_info['End_Time'] = (datetime.now() + timedelta(minutes=1)).time()
                    
            return ground_truth, gps_df, protocol_info
                            
        except Exception as e:
            logging.error(f"Error processing player data: {str(e)}")
            logging.error(traceback.format_exc())
            # Return a minimal valid protocol_info dictionary
            default_protocol_info = {
                'Start_Time': datetime.now().time(),
                'End_Time': (datetime.now() + timedelta(minutes=1)).time(),
                'Protocol ID': protocol_id,
                'hardware_type': hardware_type
            }
            return pd.DataFrame(), pd.DataFrame(), default_protocol_info

    def load_xseed_data(self, player: str, start_time: time, end_time: time, 
                time_tolerance: int = 0, hardware_type: str = 'old') -> pd.DataFrame:
        """Load Xseed data for a specific hardware type with consistent sampling."""
        try:
            # Convert time parameters to time objects if they're not already
            if not isinstance(start_time, time):
                try:
                    start_time = pd.to_datetime(start_time).time()
                except Exception as e:
                    logging.warning(f"Failed to convert start_time: {e}")
                    start_time = datetime.now().time()
                    
            if not isinstance(end_time, time):
                try:
                    end_time = pd.to_datetime(end_time).time()
                except Exception as e:
                    logging.warning(f"Failed to convert end_time: {e}")
                    end_time = (datetime.now() + timedelta(minutes=1)).time()
            
            # Determine file path based on hardware type
            player_num = player.split()[-1]
            player_dir = f'player_{player_num.replace(".", "_")}'
            
            file_path = None
            if hardware_type == 'old':
                # Check for specific old hardware file
                for ext in ['.csv', '.xlsx', '.xls']:
                    old_file = self.base_path / f'{player_dir}/df_trace_old{ext}'
                    if old_file.exists():
                        file_path = old_file
                        break
                if not file_path:
                    # Try standard file as fallback
                    for ext in ['.csv', '.xlsx', '.xls']:
                        std_file = self.base_path / f'{player_dir}/df_trace{ext}'
                        if std_file.exists():
                            file_path = std_file
                            break
            else:  # new hardware
                # Check for specific new hardware file in regular player directory
                for ext in ['.csv', '.xlsx', '.xls']:
                    new_file = self.base_path / f'{player_dir}/df_trace_new{ext}'
                    if new_file.exists():
                        file_path = new_file
                        break
                if not file_path:
                    # Try in decimal player directory
                    decimal_num = f"{player_num}.5"
                    decimal_dir = f'player_{decimal_num.replace(".", "_")}'
                    for ext in ['.csv', '.xlsx', '.xls']:
                        decimal_file = self.base_path / f'{decimal_dir}/df_trace{ext}'
                        if decimal_file.exists():
                            file_path = decimal_file
                            break
            
            # If no file was found, log and return empty DataFrame
            if not file_path or not file_path.exists():
                logging.warning(f"No Xseed file found for {player} with {hardware_type} hardware")
                return pd.DataFrame()
                    
            logging.info(f"Loading Xseed data from: {file_path}")
            
            # Load file based on extension
            try:
                if str(file_path).endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    # Try multiple CSV parsing approaches
                    try:
                        df = pd.read_csv(file_path, sep=",")
                        
                        # Check if parsing looks correct
                        if len(df.columns) <= 2:
                            df = pd.read_csv(file_path, sep=";")
                            
                            if len(df.columns) <= 2:
                                df = pd.read_csv(file_path, sep=None, engine='python')
                    except:
                        # Last resort
                        df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception as load_error:
                logging.error(f"Error loading Xseed data: {str(load_error)}")
                return pd.DataFrame()
                
            # Debug log columns and data sample
            logging.info(f"Xseed columns: {df.columns.tolist()}")
            logging.info(f"Xseed data sample: {df.head(2)}")
            
            # Find timestamp column
            time_cols = ['gps_time', 'time', 'timestamp', 'date', 'datetime', 'date time']
            time_col = None
            
            for col in df.columns:
                if any(time_name in col.lower() for time_name in time_cols):
                    time_col = col
                    break
                    
            if not time_col:
                logging.error(f"No timestamp column found in Xseed data")
                return pd.DataFrame()
                
            logging.info(f"Using {time_col} as timestamp column")
            
            # Convert timestamps to datetime
            try:
                # Try with direct parsing first
                df['date time'] = pd.to_datetime(df[time_col], errors='coerce')
                
                # Drop rows with invalid timestamps
                df = df.dropna(subset=['date time'])
                
                if df.empty:
                    logging.error(f"No valid timestamps in Xseed data")
                    return pd.DataFrame()
                    
                # Extract time of day
                df['time_of_day'] = df['date time'].dt.time
                
            except Exception as time_error:
                logging.error(f"Error processing timestamps: {str(time_error)}")
                return pd.DataFrame()
                
            # Handle time tolerance for filtering
            try:
                time_start = start_time
                time_end = end_time
                
                # Apply time tolerance
                if time_tolerance < 0:
                    time_start = (datetime.combine(datetime.today(), time_start) + 
                                timedelta(seconds=time_tolerance)).time()
                elif time_tolerance > 0:
                    time_end = (datetime.combine(datetime.today(), time_end) + 
                            timedelta(seconds=time_tolerance)).time()
                            
                logging.info(f"Filtering Xseed data for time window: {time_start} to {time_end}")
                
                # Filter by time window
                time_mask = (df['time_of_day'] >= time_start) & (df['time_of_day'] <= time_end)
                filtered_df = df[time_mask].copy()
                
                logging.info(f"Found {len(filtered_df)} points in time window for {player} with {hardware_type} hardware")
            except Exception as filter_error:
                logging.error(f"Error filtering by time window: {str(filter_error)}")
                return pd.DataFrame()
            
            if filtered_df.empty:
                logging.warning(f"No data points in time window for {player} with {hardware_type} hardware")
                return pd.DataFrame()
                
            # Process coordinates
            try:
                # Check for x/y coordinates
                has_xy = False
                
                # Try standard column names, prioritizing x_ma_3 and y_ma_3
                x_cols = ['x_ma_3', 'x', 'x_position', 'pos_x']
                y_cols = ['y_ma_3', 'y', 'y_position', 'pos_y']
                
                x_col = next((col for col in x_cols if col in filtered_df.columns), None)
                y_col = next((col for col in y_cols if col in filtered_df.columns), None)
                
                if x_col and y_col:
                    has_xy = True
                    logging.info(f"Using existing x/y coordinates: {x_col}, {y_col}")
                    filtered_df['x_real_m'] = filtered_df[x_col]
                    filtered_df['y_real_m'] = filtered_df[y_col]
                else:
                    # Check for latitude/longitude columns
                    lat_cols = ['lat', 'latitude', 'gps_lat']
                    lng_cols = ['lng', 'long', 'longitude', 'gps_lng']
                    
                    lat_col = next((col for col in lat_cols if col in filtered_df.columns), None)
                    lng_col = next((col for col in lng_cols if col in filtered_df.columns), None)
                    
                    if lat_col and lng_col:
                        logging.info(f"Converting GPS coordinates to local coordinates")
                        
                        # Transform GPS coordinates to local coordinates
                        transformed_data = []
                        origin_point = {"lat": self.corner1_lat, "lng": self.corner1_lon}
                        
                        for _, row in filtered_df.iterrows():
                            if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
                                continue
                                
                            target_point = {"lat": row[lat_col], "lng": row[lng_col]}
                            result = HaversineService.distance_haversine(origin_point, target_point, meters=1000,rotation=self.pitch_service.rotation)
                            
                            if result["dx"] is not None and result["dy"] is not None:
                                row_dict = row.to_dict()
                                row_dict['x_real_m'] = result["dx"]
                                row_dict['y_real_m'] = result["dy"]
                                transformed_data.append(row_dict)
                        
                        if transformed_data:
                            filtered_df = pd.DataFrame(transformed_data)
                            has_xy = True
                            logging.info(f"Successfully transformed {len(filtered_df)} GPS points to local coordinates")
                        else:
                            logging.warning(f"No valid coordinate transformations for {player}")
                            return pd.DataFrame()
                    else:
                        logging.error(f"No coordinate columns found in Xseed data")
                        return pd.DataFrame()
                        
                # Check for speed column
                speed_cols = ['speed', 'gps_speed', 'gps_speed_mean', 'velocity']
                speed_col = next((col for col in speed_cols if col in filtered_df.columns), None)
                
                if speed_col:
                    filtered_df['speed'] = filtered_df[speed_col]
                else:
                    # Default speed = 0
                    filtered_df['speed'] = 0
                    
                # Add hardware type to Xseed data
                filtered_df['hardware_type'] = hardware_type
                    
                # Calculate original sampling rate
                if len(filtered_df) > 1:
                    xseed_time_range = (filtered_df['date time'].max() - filtered_df['date time'].min()).total_seconds()
                    xseed_sampling_rate = len(filtered_df) / xseed_time_range if xseed_time_range > 0 else 0
                    logging.info(f"XSEED original sampling rate: {xseed_sampling_rate:.2f} Hz")
                
                # Resample to 10Hz before returning
                if has_xy and not filtered_df.empty:
                    logging.info(f"Successfully processed {len(filtered_df)} Xseed points for {player} with {hardware_type} hardware")
                    
                    # Resample to 10Hz before returning
                    filtered_df = self.data_sync.resample_data(filtered_df, target_frequency='100ms')
                    logging.info(f"Resampled Xseed data to 10Hz: {len(filtered_df)} points")
                    
                    return filtered_df
                else:
                    logging.warning(f"Failed to process Xseed data for {player} with {hardware_type} hardware")
                    return pd.DataFrame()
                    
            except Exception as e:
                logging.error(f"Error processing Xseed coordinates: {str(e)}")
                logging.error(traceback.format_exc())
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error loading Xseed data for {player} with {hardware_type} hardware: {str(e)}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    def _find_file(self, base_name: str) -> Path:
        """
        Find file with given base name, checking for both CSV and Excel formats.
        Returns the path to the file that exists, prioritizing CSV over Excel.
        """
        csv_path = self.base_path / f'{base_name}.csv'
        xlsx_path = self.base_path / f'{base_name}.xlsx'
        xls_path = self.base_path / f'{base_name}.xls'
        
        if csv_path.exists():
            return csv_path
        elif xlsx_path.exists():
            return xlsx_path
        elif xls_path.exists():
            return xls_path
        else:
            # Return the CSV path as default even if it doesn't exist
            return csv_path
    def _detect_available_players(self) -> List[str]:
        """
        Detect available players in the session by looking for player directories.
        Returns a list of player identifiers like ['Player 1', 'Player 2', ...]
        """
        players = []
        
        # Look for player_N directories
        for i in range(1, 7):  # Check for up to 6 players
            player_dir = self.base_path / f'player_{i}'
            if player_dir.exists() and player_dir.is_dir():
                players.append(f'Player {i}')
        
        # If no players found, default to 3 players
        if not players:
            logging.warning(f"No player directories detected in {self.session_id}, defaulting to 3 players")
            players = ['Player 1', 'Player 2', 'Player 3']
        
        return sorted(players, key=self._player_sort_key)
    def get_available_hardware_types(self, player: str) -> List[str]:
        """Get list of available hardware types for a specific player."""
        available = []
        
        # Check for old hardware
        old_hw_indicators = [
            self._check_file_exists(f'Protocol_Sheet_{player[-1]}'),
            self._check_file_exists(f'player_{player[-1].replace(".", "_")}/df_trace_old'),
            self._check_file_exists(f'player_{player[-1].replace(".", "_")}/df_trace')
        ]
        if any(old_hw_indicators):
            available.append('old')
        
        # Check for new hardware
        decimal_num = f"{player[-1]}.5"
        new_hw_indicators = [
            self._check_file_exists(f'Protocol_Sheet_{decimal_num}'),
            self._check_file_exists(f'player_{player[-1].replace(".", "_")}/df_trace_new'),
            self._check_file_exists(f'player_{decimal_num.replace(".", "_")}/df_trace')
        ]
        if any(new_hw_indicators):
            available.append('new')
        
        return available
    def _check_file_exists(self, relative_path: str, any_extension=True) -> bool:
        """Check if file exists with any of the supported extensions."""
        if any_extension:
            for ext in ['.csv', '.xlsx', '.xls']:
                if (self.base_path / f"{relative_path}{ext}").exists():
                    return True
            return False
        else:
            return (self.base_path / relative_path).exists()

    def _player_sort_key(self, player_str: str) -> float:
        """Create a sort key to order players numerically, handling non-integer numbers."""
        # Extract number from player string (e.g., 'Player 2.5' -> 2.5)
        try:
            return float(player_str.split()[-1])
        except (ValueError, IndexError):
            return float('inf')  # Put invalid players at the end


    def _detect_xseed_version(self) -> str:
        """
        Detect which Xseed hardware version to use based on file availability.
        Returns "new" if new hardware files exist for any player, otherwise "old"
        """
        # Check all player directories for new hardware files
        for player in self.players:
            player_num = player.split()[-1]
            player_dir = f'player_{player_num.replace(".", "_")}'
            
            # Check for new hardware file with any supported extension
            new_file_exists = any(
                (self.base_path / f'{player_dir}/df_trace_new{ext}').exists()
                for ext in ['.csv', '.xlsx', '.xls']
            )
            
            if new_file_exists:
                logging.info(f"Detected new Xseed hardware for {player} in session {self.session_id}")
                return "new"
        
        # If no new hardware files found for any player, default to old hardware
        logging.info(f"Using old Xseed hardware for session {self.session_id}")
        return "old"

    def calculate_player_metrics(self, gt_df: pd.DataFrame, 
                               gps_df: pd.DataFrame,
                               time_tolerance: int) -> Dict:
        """
        Calculate metrics for player tracking accuracy.
        
        Args:
            gt_df: Ground truth data
            gps_df: GPS tracking data
            time_tolerance: Time tolerance in seconds
            
        Returns:
            Dictionary containing various accuracy metrics
        """
        all_errors = []
        activity_errors = {}

        # Calculate errors for each activity
        for activity_id in gt_df['Activity_ID'].unique():
            gt_activity = gt_df[gt_df['Activity_ID'] == activity_id]
            activity_errors[activity_id] = []
            
            for _, gps_point in gps_df.iterrows():
                relevant_gt = gt_activity[
                    abs(gt_activity['date time'] - gps_point['date time']) <= 
                    pd.Timedelta(seconds=time_tolerance)
                ]
                
                if not relevant_gt.empty:
                    distances = np.sqrt(
                        (relevant_gt['x_real_m'] - gps_point['x_real_m'])**2 +
                        (relevant_gt['y_real_m'] - gps_point['y_real_m'])**2
                    )
                    min_error = float(np.min(distances))
                    activity_errors[activity_id].append(min_error)
                    all_errors.append(min_error)

            # Calculate metrics per activity
            if activity_errors[activity_id]:
                errors = np.array(activity_errors[activity_id])
                activity_errors[activity_id] = {
                    'mean_error': float(np.mean(errors)),
                    'max_error': float(np.max(errors)),
                    'rmse': float(np.sqrt(np.mean(np.square(errors)))),
                    'std_error': float(np.std(errors)),
                    'number_of_points': len(errors)
                }

        # Calculate overall metrics
        if all_errors:
            errors = np.array(all_errors)
            overall_metrics = {
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'std_error': float(np.std(errors)),
                'number_of_points': len(errors)
            }
        else:
            overall_metrics = {
                'mean_error': np.nan,
                'max_error': np.nan,
                'rmse': np.nan,
                'std_error': np.nan,
                'number_of_points': 0
            }

        return {
            'overall': overall_metrics,
            'by_activity': activity_errors
        }

    def calculate_velocity_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate speed-related metrics from tracking data.
        
        Args:
            data: DataFrame containing speed data
            
        Returns:
            Dictionary of velocity metrics
        """
        try:
            if 'speed' not in data.columns:
                return {}
                
            return {
                'mean_speed': float(data['speed'].mean()),
                'max_speed': float(data['speed'].max()),
                'std_speed': float(data['speed'].std()),
                'total_distance': float(np.sum(data['speed'] * 0.1))  # Assuming 10Hz sampling
            }
            
        except Exception as e:
            logging.error(f"Error calculating velocity metrics: {str(e)}")
            return {}

    def prepare_shinguard_data(self, shinguard_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare shinguard data for analysis by standardizing format.
        
        Args:
            shinguard_data: Raw shinguard data
            
        Returns:
            Processed DataFrame with standardized columns
        """
        # Make a copy to avoid modifying the original
        data = shinguard_data.copy()
        
        # Rename columns to standard format
        if 'gps_time' in data.columns:
            data = data.rename(columns={
                'gps_time': 'date time',
                'gps_speed': 'shin_speed',
                'gps_direction': 'shin_direction'
            })
        
        # Convert lat/lng to latitude/longitude if needed
        if 'lat' in data.columns and 'lng' in data.columns:
            data = data.rename(columns={
                'lat': 'latitude',
                'lng': 'longitude'
            })
        
        # Add required columns if missing
        if 'impact' not in data.columns:
            # Calculate synthetic impact from speed changes
            data['impact'] = data['shin_speed'].diff().abs().fillna(0)
        
        if 'z' not in data.columns:
            data['z'] = 0.0  # Add default Z coordinate
            
        return data

    def calculate_coverage_metrics(self, 
                                 gt_data: pd.DataFrame, 
                                 tracking_data: pd.DataFrame) -> Dict:
        """
        Calculate temporal and spatial coverage metrics.
        
        Args:
            gt_data: Ground truth data
            tracking_data: Tracking system data
            
        Returns:
            Dictionary containing coverage metrics
        """
        try:
            if gt_data.empty or tracking_data.empty:
                return {
                    'temporal_coverage': 0.0,
                    'spatial_coverage': 0.0,
                    'total_time': 0.0,
                    'covered_time': 0.0
                }
            
            # Calculate temporal coverage
            total_time = (gt_data['date time'].max() - 
                         gt_data['date time'].min()).total_seconds()
            tracked_time = (tracking_data['date time'].max() - 
                          tracking_data['date time'].min()).total_seconds()
            temporal_coverage = tracked_time / total_time if total_time > 0 else 0.0
            
            # Calculate spatial coverage (using grid-based approach)
            grid_size = 1.0  # meters per grid cell
            gt_cells = set(zip(
                (gt_data['x_real_m'] // grid_size).astype(int),
                (gt_data['y_real_m'] // grid_size).astype(int)
            ))
            track_cells = set(zip(
                (tracking_data['x_real_m'] // grid_size).astype(int),
                (tracking_data['y_real_m'] // grid_size).astype(int)
            ))
            
            spatial_coverage = len(track_cells) / len(gt_cells) if gt_cells else 0.0
            
            return {
                'temporal_coverage': temporal_coverage,
                'spatial_coverage': spatial_coverage,
                'total_time': total_time,
                'covered_time': tracked_time
            }
            
        except Exception as e:
            logging.error(f"Error calculating coverage metrics: {str(e)}")
            return {
                'temporal_coverage': 0.0,
                'spatial_coverage': 0.0,
                'total_time': 0.0,
                'covered_time': 0.0
            }

    def calculate_distance(self, 
                         point1: Tuple[float, float], 
                         point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points in meters.
        
        Args:
            point1: First point (x, y) coordinates
            point2: Second point (x, y) coordinates
            
        Returns:
            Distance in meters
        """
        return np.sqrt(
            (point2[0] - point1[0])**2 + 
            (point2[1] - point1[1])**2
        )

    def interpolate_position(self, 
                           time: datetime, 
                           prev_point: Dict, 
                           next_point: Dict) -> Tuple[float, float]:
        """
        Linearly interpolate position between two points at given time.
        
        Args:
            time: Target time for interpolation
            prev_point: Previous position data point
            next_point: Next position data point
            
        Returns:
            Tuple of (x, y) interpolated coordinates
        """
        try:
            # Calculate time fractions
            total_time = (next_point['date time'] - prev_point['date time']).total_seconds()
            if total_time == 0:
                return prev_point['x_real_m'], prev_point['y_real_m']
                
            frac = (time - prev_point['date time']).total_seconds() / total_time
            
            # Interpolate x and y coordinates
            x = prev_point['x_real_m'] + frac * (next_point['x_real_m'] - prev_point['x_real_m'])
            y = prev_point['y_real_m'] + frac * (next_point['y_real_m'] - prev_point['y_real_m'])
            
            return x, y
            
        except Exception as e:
            logging.error(f"Error interpolating position: {str(e)}")
            return None