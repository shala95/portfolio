import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import traceback
from dataclasses import dataclass
from src.utils import load_file
from datetime import datetime, timedelta
import services.haversine_service as haversine_service
import services.pitch_service as pitch_service
import pandas as pd
import numpy as np
import os
import logging
import traceback
from typing import List, Dict, Optional
from datetime import datetime, timedelta, time
from pathlib import Path
from services.pitch_service import PitchService
@dataclass
class ProtocolSummary:
    """Data class for protocol summary information."""
    protocol_id: int
    name: str
    start_time: datetime.time
    end_time: datetime.time
    multi_analysis: bool
    total_activities: int
    total_distance: float
    players: Set[str]
    synchronized: bool

class GroundTruthProcessor:
    """
    Process and manage ground truth data from protocols and activity sheets.
    Handles data loading, coordination transformation, and path generation.
    """
    
    def __init__(self, protocol_path: str, activity_paths: List[str], pitch_service: PitchService,session_id: str):
        self.protocol_path = protocol_path
        self.activity_paths = activity_paths
        self.pitch_service = pitch_service
        self.session_id = session_id
        # Try to find the protocol file with any supported extension
        protocol_file = self._find_file_with_any_extension(protocol_path)
        if protocol_file:
            self.protocols_df = self.load_protocols(protocol_file)
        else:
            logging.warning(f"No protocol file found for path: {protocol_path}")
            self.protocols_df = pd.DataFrame()
        
        # Load activity data
        self.activities_df = self.load_activity_files(activity_paths)

    def load_protocols(self, protocol_file: str) -> pd.DataFrame:
        """
        Load protocol data from file with enhanced encoding detection.
        Tries multiple encodings and handling for malformed files.
        """
        logging.info(f"Loading protocol file: {protocol_file}")
        
        # Check if file exists
        if not os.path.exists(protocol_file):
            logging.error(f"Protocol file not found: {protocol_file}")
            return pd.DataFrame()
        
        # Log the first bytes of the file to help with debugging
        try:
            with open(protocol_file, 'rb') as f:
                first_bytes = f.read(200)
                logging.info(f"First bytes of protocol file: {first_bytes}")
        except:
            pass
            
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        separators = [';', ',', '\t']
        
        # Try different approaches in priority order
        
        # 1. First try with each encoding and separator combination
        for encoding in encodings:
            for sep in separators:
                try:
                    # Try to read the file with the current encoding and separator
                    df = pd.read_csv(protocol_file, sep=sep, encoding=encoding)
                    
                    # Check if df has reasonable columns (more than 1)
                    if len(df.columns) <= 1:
                        continue
                    
                    # Successfully read the file
                    logging.info(f"Successfully loaded protocol file with encoding {encoding} and separator {sep}")
                    logging.info(f"Columns: {df.columns.tolist()}")
                    
                    # Fix column names that might have trailing whitespace
                    logging.info("Cleaning column names (removing whitespace)")
                    orig_columns = df.columns.tolist()
                    df.columns = df.columns.str.strip()
                    clean_columns = df.columns.tolist()

                    # Log changes for debugging
                    changed_columns = {orig: clean for orig, clean in zip(orig_columns, clean_columns) if orig != clean}
                    if changed_columns:
                        logging.info(f"Cleaned whitespace from column names: {changed_columns}")
                        logging.info(f"Final columns: {clean_columns}")
                    
                    # Process protocol data
                    if 'Protocol ID' not in df.columns:
                        # Look for variant column names
                        protocol_id_col = None
                        for col in df.columns:
                            if 'protocol' in col.lower() and 'id' in col.lower():
                                protocol_id_col = col
                                break
                        
                        if protocol_id_col:
                            df = df.rename(columns={protocol_id_col: 'Protocol ID'})
                    
                    # Fix Protocol name column
                    if 'Protocol name' not in df.columns:
                        # Look for variant column names
                        protocol_name_col = None
                        for col in df.columns:
                            if 'protocol' in col.lower() and 'name' in col.lower():
                                protocol_name_col = col
                                break
                        
                        if protocol_name_col:
                            df = df.rename(columns={protocol_name_col: 'Protocol name'})
                        elif 'Protocol ID' in df.columns:
                            # Create a default name based on ID
                            df['Protocol name'] = 'Protocol ' + df['Protocol ID'].astype(str)
                    
                    # Fix time columns
                    for time_col, new_name in [('start_time', 'Start_Time'), ('end_time', 'End_Time')]:
                        if new_name not in df.columns:
                            # Look for variant column names
                            col_to_rename = None
                            for col in df.columns:
                                if time_col in col.lower() or new_name.lower() in col.lower():
                                    col_to_rename = col
                                    break
                            
                            if col_to_rename:
                                df = df.rename(columns={col_to_rename: new_name})
                    
                    # Fix 1: Better time parsing with explicit format
                    time_formats = ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M %p']
                    
                    for time_col in ['Start_Time', 'End_Time']:
                        if time_col in df.columns:
                            # Skip if already a time object
                            if not df.empty and isinstance(df[time_col].iloc[0], datetime.time):
                                continue
                            
                            # Try each format
                            success = False
                            for time_format in time_formats:
                                try:
                                    df[time_col] = pd.to_datetime(df[time_col], format=time_format, errors='coerce').dt.time
                                    # Check if we got valid times
                                    if not df[time_col].isna().all():
                                        success = True
                                        logging.info(f"Successfully parsed {time_col} using format {time_format}")
                                        break
                                except:
                                    continue
                            
                            # If all formats failed, try flexible parsing
                            if not success:
                                try:
                                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.time
                                    logging.info(f"Parsed {time_col} using flexible datetime parsing")
                                except:
                                    logging.warning(f"Could not parse {time_col} in protocol file")
                    
                    # Fix multi_analysis column handling
                    if 'multi_analysis' not in df.columns:
                        df['multi_analysis'] = True  # Default all protocols to multi-analysis capable
                    else:
                        # For Protocol_Sheet_1.xlsx format where multi_analysis is numeric (0/1)
                        try:
                            # First ensure it's numeric
                            df['multi_analysis'] = pd.to_numeric(df['multi_analysis'], errors='coerce').fillna(0)
                            
                            # Keep as numeric 0/1 since that's the actual format in the file
                            # This way == 0 and != 0 comparisons will work correctly
                            logging.info(f"Successfully parsed multi_analysis as numeric 0/1 values")
                            logging.info(f"multi_analysis unique values: {df['multi_analysis'].unique()}")
                        except Exception as e:
                            logging.warning(f"Error converting multi_analysis to numeric: {str(e)}")
                            # Fall back to string-based mapping approach
                            try:
                                # Convert to string first then map
                                df['multi_analysis'] = df['multi_analysis'].astype(str).map(
                                    {'1': True, 'True': True, 'true': True, 
                                    '0': False, 'False': False, 'false': False}
                                )
                                df['multi_analysis'] = df['multi_analysis'].fillna(True)  # Default to True
                                logging.info(f"Converted multi_analysis to boolean via string mapping")
                            except Exception as e2:
                                logging.warning(f"Error converting multi_analysis to boolean: {str(e2)}")
                                # Last resort: force to True
                                df['multi_analysis'] = True
                    
                    # Log a sample of the loaded data
                    if not df.empty:
                        logging.info(f"Protocol sample: {df.iloc[0].to_dict()}")
                    
                    return df
                    
                except Exception as e:
                    # Log the error but continue trying other combinations
                    logging.debug(f"Failed with encoding={encoding}, sep={sep}: {str(e)}")
                    continue
        
        # 2. If CSV approaches failed, try Excel
        if protocol_file.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(protocol_file)
                logging.info(f"Successfully loaded protocol file as Excel")
                
                # Fix column names that might have trailing whitespace
                logging.info("Cleaning column names (removing whitespace)")
                orig_columns = df.columns.tolist()
                df.columns = df.columns.str.strip()
                clean_columns = df.columns.tolist()

                # Log changes for debugging
                changed_columns = {orig: clean for orig, clean in zip(orig_columns, clean_columns) if orig != clean}
                if changed_columns:
                    logging.info(f"Cleaned whitespace from column names: {changed_columns}")
                    logging.info(f"Final columns: {clean_columns}")
                
                # Process columns as above
                if 'Protocol ID' not in df.columns:
                    for col in df.columns:
                        if 'protocol' in col.lower() and 'id' in col.lower():
                            df = df.rename(columns={col: 'Protocol ID'})
                
                # Process time fields
                for time_col in ['Start_Time', 'End_Time']:
                    if time_col in df.columns and not df.empty:
                        try:
                            # Use this safer approach without direct isinstance check
                            if pd.api.types.is_datetime64_dtype(df[time_col]):
                                df[time_col] = df[time_col].dt.time
                            else:
                                df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.time
                        except Exception as te:
                            logging.warning(f"Error converting time column {time_col}: {str(te)}")
                
                # Fix multi_analysis column handling
                if 'multi_analysis' not in df.columns:
                    df['multi_analysis'] = True  # Default all protocols to multi-analysis capable
                else:
                    # For Protocol_Sheet_1.xlsx format where multi_analysis is numeric (0/1)
                    try:
                        # First ensure it's numeric
                        df['multi_analysis'] = pd.to_numeric(df['multi_analysis'], errors='coerce').fillna(0)
                        
                        # Keep as numeric 0/1 since that's the actual format in the file
                        logging.info(f"Successfully parsed multi_analysis as numeric 0/1 values from Excel")
                        logging.info(f"multi_analysis unique values: {df['multi_analysis'].unique()}")
                    except Exception as e:
                        logging.warning(f"Error converting multi_analysis to numeric: {str(e)}")
                        # Fall back to boolean conversion - FIX THE ISINSTANCE ERROR HERE
                        try:
                            # Don't use isinstance() directly on DataFrame column values
                            # Instead, use a safer conversion approach
                            df['multi_analysis'] = df['multi_analysis'].astype(str).map({'1': True, 'True': True, 'true': True, '0': False, 'False': False, 'false': False})
                            df['multi_analysis'] = df['multi_analysis'].fillna(True)  # Default unknown values to True
                            logging.info(f"Converted multi_analysis to boolean using string mapping")
                        except Exception as e2:
                            logging.warning(f"Error converting multi_analysis to boolean: {str(e2)}")
                            # Last resort: force to True
                            df['multi_analysis'] = True
                
                return df
                
            except Exception as ex:
                logging.error(f"Failed to load Excel protocol file: {str(ex)}")
                logging.error(f"Exception details: {traceback.format_exc()}")  # Add detailed traceback
        
        # 3. Try CSV with Python engine (more flexible)
        try:
            df = pd.read_csv(protocol_file, sep=None, engine='python')
            
            # Fix column names that might have trailing whitespace
            logging.info("Cleaning column names (removing whitespace)")
            orig_columns = df.columns.tolist()
            df.columns = df.columns.str.strip()
            clean_columns = df.columns.tolist()

            # Log changes for debugging
            changed_columns = {orig: clean for orig, clean in zip(orig_columns, clean_columns) if orig != clean}
            if changed_columns:
                logging.info(f"Cleaned whitespace from column names: {changed_columns}")
                logging.info(f"Final columns: {clean_columns}")
            
            # Process columns as above
            if 'Protocol ID' not in df.columns:
                for col in df.columns:
                    if 'protocol' in col.lower() and 'id' in col.lower():
                        df = df.rename(columns={col: 'Protocol ID'})
            
            # Process time fields
            for time_col in ['Start_Time', 'End_Time']:
                if time_col in df.columns and not df.empty:
                    try:
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.time
                    except:
                        pass
            
            # Fix multi_analysis column handling
            if 'multi_analysis' not in df.columns:
                df['multi_analysis'] = True  # Default all protocols to multi-analysis capable
            else:
                try:
                    # First ensure it's numeric
                    df['multi_analysis'] = pd.to_numeric(df['multi_analysis'], errors='coerce').fillna(0)
                    logging.info(f"Successfully parsed multi_analysis as numeric with Python engine")
                except Exception as e:
                    logging.warning(f"Error converting multi_analysis to numeric: {str(e)}")
                    # Fall back to string-based mapping approach
                    df['multi_analysis'] = df['multi_analysis'].astype(str).map(
                        {'1': True, 'True': True, 'true': True, 
                        '0': False, 'False': False, 'false': False}
                    )
                    df['multi_analysis'] = df['multi_analysis'].fillna(True)  # Default to True
            
            return df
            
        except Exception as ex:
            logging.error(f"All attempts to load protocol file failed: {str(ex)}")
            logging.error(f"Exception details: {traceback.format_exc()}")
        
        # 4. Last resort: try to create the structure from a raw read
        try:
            with open(protocol_file, 'r', errors='replace') as f:
                lines = f.readlines()
                
            if lines:
                # Try to detect header and separator
                header = lines[0].strip()
                
                if ';' in header:
                    sep = ';'
                elif ',' in header:
                    sep = ','
                else:
                    sep = None
                
                if sep:
                    columns = header.split(sep)
                    # Clean column names
                    columns = [col.strip() for col in columns]
                    data = []
                    
                    for line in lines[1:]:
                        if line.strip():  # Skip empty lines
                            values = line.strip().split(sep)
                            # Pad with None if needed
                            values += [None] * (len(columns) - len(values))
                            data.append(values[:len(columns)])  # Trim if too long
                    
                    df = pd.DataFrame(data, columns=columns)
                    
                    # Create minimal required columns
                    if 'Protocol ID' not in df.columns:
                        df['Protocol ID'] = range(1, len(df) + 1)
                    
                    if 'Protocol name' not in df.columns:
                        df['Protocol name'] = 'Protocol ' + df['Protocol ID'].astype(str)
                    
                    # Add multi_analysis with default value
                    if 'multi_analysis' not in df.columns:
                        df['multi_analysis'] = 1  # Default to 1 (enabled) for all protocols
                    
                    return df
        except Exception as e:
            logging.error(f"Raw line-by-line loading failed: {str(e)}")
            logging.error(f"Exception details: {traceback.format_exc()}")
        
        # If everything fails, return an empty DataFrame
        logging.error(f"Failed to load protocol file {protocol_file}")
        return pd.DataFrame()

    def load_activities(self, activity_file: str) -> pd.DataFrame:
        """
        Load activity data from file with enhanced encoding detection.
        """
        logging.info(f"Loading activity file: {activity_file}")
        
        # Check if file exists
        if not os.path.exists(activity_file):
            logging.error(f"Activity file not found: {activity_file}")
            return pd.DataFrame()
        
        # Log the first bytes of the file to help with debugging
        try:
            with open(activity_file, 'rb') as f:
                first_bytes = f.read(200)
                logging.info(f"First bytes of activity file: {first_bytes}")
        except:
            pass
        
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        separators = [';', ',', '\t']
        
        # Try different approaches in priority order
        
        # 1. First try with each encoding and separator combination
        for encoding in encodings:
            for sep in separators:
                try:
                    # Try to read the file with the current encoding and separator
                    df = pd.read_csv(activity_file, sep=sep, encoding=encoding)
                    
                    # Check if df has reasonable columns (more than 1)
                    if len(df.columns) <= 1:
                        continue
                    
                    # Successfully read the file
                    logging.info(f"Successfully loaded activity file with encoding {encoding} and separator {sep}")
                    logging.info(f"Columns: {df.columns.tolist()}")
                    
                    # Process activity data - standardize column names
                    column_mappings = {}
                    
                    # Check for Activity_ID
                    if 'Activity_ID' not in df.columns:
                        for col in df.columns:
                            if 'activity' in col.lower() and 'id' in col.lower():
                                column_mappings[col] = 'Activity_ID'
                                break
                    
                    # Check for Protocol_ID
                    if 'Protocol_ID' not in df.columns:
                        for col in df.columns:
                            if 'protocol' in col.lower() and 'id' in col.lower():
                                column_mappings[col] = 'Protocol_ID'
                                break
                    
                    # Check for start/end coordinates
                    coord_mappings = {
                        'Start_coordinates': ['start_coordinates', 'start_coords', 'starting_coordinates'],
                        'End_Coordinates': ['end_coordinates', 'end_coords', 'ending_coordinates']
                    }
                    
                    for target, variants in coord_mappings.items():
                        if target not in df.columns:
                            for variant in variants:
                                if any(variant in col.lower() for col in df.columns):
                                    for col in df.columns:
                                        if variant in col.lower():
                                            column_mappings[col] = target
                                            break
                                    break
                    
                    # Check for Activity_name
                    if 'Activity_name' not in df.columns:
                        for col in df.columns:
                            if 'activity' in col.lower() and 'name' in col.lower():
                                column_mappings[col] = 'Activity_name'
                                break
                    
                    # Check for Start_Time and End_Time
                    time_mappings = {
                        'Start_Time': ['start_time', 'starting_time', 'begin_time'],
                        'End_Time': ['end_time', 'ending_time', 'finish_time']
                    }
                    
                    for target, variants in time_mappings.items():
                        if target not in df.columns:
                            for variant in variants:
                                if any(variant in col.lower() for col in df.columns):
                                    for col in df.columns:
                                        if variant in col.lower():
                                            column_mappings[col] = target
                                            break
                                    break
                    
                    # Apply mappings
                    if column_mappings:
                        df = df.rename(columns=column_mappings)
                        logging.info(f"Renamed columns: {column_mappings}")
                    
                    # Process time columns
                    time_formats = ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M %p']
                    
                    for time_col in ['Start_Time', 'End_Time']:
                        if time_col in df.columns:
                            # Skip if already a time object
                            if not df.empty and isinstance(df[time_col].iloc[0], datetime.time):
                                continue
                            
                            # Try each format
                            success = False
                            for time_format in time_formats:
                                try:
                                    df[time_col] = pd.to_datetime(df[time_col], format=time_format, errors='coerce').dt.time
                                    # Check if we got valid times
                                    if not df[time_col].isna().all():
                                        success = True
                                        logging.info(f"Successfully parsed {time_col} using format {time_format}")
                                        break
                                except:
                                    continue
                            
                            # If all formats failed, try flexible parsing
                            if not success:
                                try:
                                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.time
                                    logging.info(f"Parsed {time_col} using flexible datetime parsing")
                                except:
                                    logging.warning(f"Could not parse {time_col} in activity file")
                    
                    # Convert ID columns to numeric if possible
                    for id_col in ['Activity_ID', 'Protocol_ID']:
                        if id_col in df.columns:
                            try:
                                df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
                            except:
                                pass
                    
                    # Log a sample of the loaded data
                    if not df.empty:
                        logging.info(f"Activity sample: {df.iloc[0].to_dict()}")
                    
                    return df
                    
                except Exception as e:
                    # Log the error but continue trying other combinations
                    logging.debug(f"Failed with encoding={encoding}, sep={sep}: {str(e)}")
                    continue
        
        # 2. If CSV approaches failed, try Excel
        if activity_file.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(activity_file)
                logging.info(f"Successfully loaded activity file as Excel")
                
                # Process columns as above
                column_mappings = {}
                
                # Check for Activity_ID and Protocol_ID
                for target, keyword_pairs in [
                    ('Activity_ID', [('activity', 'id')]),
                    ('Protocol_ID', [('protocol', 'id')])
                ]:
                    if target not in df.columns:
                        for col in df.columns:
                            if all(kw in col.lower() for kw in keyword_pairs[0]):
                                column_mappings[col] = target
                                break
                
                # Apply mappings
                if column_mappings:
                    df = df.rename(columns=column_mappings)
                
                # Convert ID columns to numeric if possible
                for id_col in ['Activity_ID', 'Protocol_ID']:
                    if id_col in df.columns:
                        try:
                            df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
                        except:
                            pass
                    
                return df
                
            except Exception as ex:
                logging.error(f"Failed to load Excel activity file: {str(ex)}")
        
        # 3. Try CSV with Python engine (more flexible)
        try:
            df = pd.read_csv(activity_file, sep=None, engine='python')
            
            # Process columns as above
            column_mappings = {}
            
            # Check for Activity_ID and Protocol_ID
            for target, keyword_pairs in [
                ('Activity_ID', [('activity', 'id')]),
                ('Protocol_ID', [('protocol', 'id')])
            ]:
                if target not in df.columns:
                    for col in df.columns:
                        if all(kw in col.lower() for kw in keyword_pairs[0]):
                            column_mappings[col] = target
                            break
            
            # Apply mappings
            if column_mappings:
                df = df.rename(columns=column_mappings)
            
            # Convert ID columns to numeric if possible
            for id_col in ['Activity_ID', 'Protocol_ID']:
                if id_col in df.columns:
                    try:
                        df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
                    except:
                        pass
                
            return df
            
        except Exception as ex:
            logging.error(f"All attempts to load activity file failed: {str(ex)}")
        
        # 4. Try a raw line-by-line approach as last resort
        try:
            with open(activity_file, 'r', errors='replace') as f:
                lines = f.readlines()
                
            if lines:
                # Try to detect header and separator
                header = lines[0].strip()
                
                if ';' in header:
                    sep = ';'
                elif ',' in header:
                    sep = ','
                else:
                    sep = None
                
                if sep:
                    columns = header.split(sep)
                    data = []
                    
                    for line in lines[1:]:
                        if line.strip():  # Skip empty lines
                            values = line.strip().split(sep)
                            # Pad with None if needed
                            values += [None] * (len(columns) - len(values))
                            data.append(values[:len(columns)])  # Trim if too long
                    
                    df = pd.DataFrame(data, columns=columns)
                    
                    # Create minimal required columns
                    if 'Activity_ID' not in df.columns:
                        df['Activity_ID'] = range(1, len(df) + 1)
                    
                    if 'Protocol_ID' not in df.columns:
                        df['Protocol_ID'] = 1  # Default to protocol 1
                    
                    return df
        except Exception as e:
            logging.error(f"Raw line-by-line loading failed: {str(e)}")
        
        # If everything fails, return an empty DataFrame
        logging.error(f"Failed to load activity file {activity_file}")
        return pd.DataFrame()

    def load_activity_files(self, activity_paths: List[str]) -> pd.DataFrame:
        """Load data from multiple activity files and combine them."""
        combined_df = pd.DataFrame()
        
        if isinstance(activity_paths, list):
            for path in activity_paths:
                # Try to find the file with any extension
                activity_file = self._find_file_with_any_extension(path)
                if activity_file:
                    df = self.load_activities(activity_file)
                    if not df.empty:
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
        elif isinstance(activity_paths, str):
            activity_file = self._find_file_with_any_extension(activity_paths)
            if activity_file:
                combined_df = self.load_activities(activity_file)
        
        return combined_df

    def _find_file_with_any_extension(self, base_path: str) -> Optional[str]:
        """Find a file with any supported extension."""
        # Try the original path first
        if Path(base_path).exists():
            return base_path
            
        # Try different extensions
        base_without_ext = str(Path(base_path).with_suffix(''))
        for ext in ['.csv', '.xlsx', '.xls']:
            test_path = base_without_ext + ext
            if Path(test_path).exists():
                return test_path
                
        # No file found
        return None

    def _process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process coordinates with explicit circle detection from Activity_name."""
        coord_df = pd.DataFrame()
        
        for idx, row in df.iterrows():
            try:
                # Process start and end coordinates
                start_coords = self._parse_coordinates(row['Start_coordinates'])
                end_coords = self._parse_coordinates(row['End_Coordinates'])
                
                # Store basic coordinates
                coord_df.at[idx, 'start_lat'] = start_coords[0]
                coord_df.at[idx, 'start_lng'] = start_coords[1]
                coord_df.at[idx, 'end_lat'] = end_coords[0]
                coord_df.at[idx, 'end_lng'] = end_coords[1]
                
                # Clean activity name and check for circle indicators
                activity_name = str(row['Activity_name']).lower().strip()
                is_circular = ('circle' in activity_name or 
                            'circular' in activity_name or 
                            'loop' in activity_name)
                
                coord_df.at[idx, 'is_circular'] = is_circular
                
                if is_circular:
                    # Calculate circle parameters
                    center_lat = (start_coords[0] + end_coords[0]) / 2
                    center_lng = (start_coords[1] + end_coords[1]) / 2
                    
                    radius_lat = abs(end_coords[0] - start_coords[0]) / 2
                    radius_lng = abs(end_coords[1] - start_coords[1]) / 2
                    
                    coord_df.at[idx, 'center_lat'] = center_lat
                    coord_df.at[idx, 'center_lng'] = center_lng
                    coord_df.at[idx, 'radius_lat'] = radius_lat
                    coord_df.at[idx, 'radius_lng'] = radius_lng
                    
                    logging.info(f"Processing circular activity {row['Activity_ID']}:")
                    logging.info(f"Start coordinates: {start_coords}")
                    logging.info(f"End coordinates: {end_coords}")
                    logging.info(f"Center: ({center_lat}, {center_lng})")
                    logging.info(f"Radius: lat={radius_lat:.6f}, lng={radius_lng:.6f}")
            
            except Exception as e:
                logging.error(f"Error processing coordinates for activity {row['Activity_ID']}: {str(e)}")
                logging.error(f"Activity data: {row.to_dict()}")
                # Set default values for error cases
                coord_df.at[idx, 'is_circular'] = False
        
        return coord_df

    def _parse_coordinates(self, coord_str: str) -> Tuple[float, float]:
        """
        Parse coordinate string into tuple of floats.
        
        Args:
            coord_str: String containing coordinates
            
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            if pd.isna(coord_str):
                return (np.nan, np.nan)
            
            # Clean and split coordinates
            coords = coord_str.strip().split(',')
            return (float(coords[0].strip()), float(coords[1].strip()))
            
        except Exception as e:
            logging.warning(f"Error parsing coordinates {coord_str}: {str(e)}")
            return (np.nan, np.nan)

    def get_ground_truth_path(self, protocol_id: int) -> pd.DataFrame:
        """
        Generate ground truth path for a protocol.
        """
        try:
            logging.info(f"Generating ground truth path for protocol {protocol_id}")
            
            # Check if we should use Protocol_ID or Protocol ID
            protocol_id_col = 'Protocol_ID' if 'Protocol_ID' in self.activities_df.columns else 'Protocol ID'
            
            # Verify protocol exists
            if self.protocols_df.empty:
                logging.error(f"No protocols loaded")
                return pd.DataFrame()
                
            protocol_rows = self.protocols_df[self.protocols_df['Protocol ID'] == protocol_id]
            if protocol_rows.empty:
                # Try with string conversion if needed
                protocol_rows = self.protocols_df[self.protocols_df['Protocol ID'] == str(protocol_id)]
                
            if protocol_rows.empty:
                logging.error(f"Protocol ID {protocol_id} not found")
                logging.info(f"Available Protocol IDs: {self.protocols_df['Protocol ID'].tolist()}")
                return pd.DataFrame()
                
            protocol_info = protocol_rows.iloc[0]
            logging.info(f"Found protocol: {protocol_info['Protocol name']}")
            logging.info(f"Time range: {protocol_info['Start_Time']} to {protocol_info['End_Time']}")
            
            # Convert protocol_id to numeric for comparison
            try:
                # Try to convert Protocol ID column to numeric
                self.activities_df[protocol_id_col] = pd.to_numeric(self.activities_df[protocol_id_col], errors='ignore')
                
                # Then convert protocol_id to numeric too
                protocol_id_numeric = pd.to_numeric(protocol_id) if not isinstance(protocol_id, (int, float)) else protocol_id
                
                # Now compare using the converted values
                activities = self.activities_df[self.activities_df[protocol_id_col] == protocol_id_numeric]
                
                logging.info(f"Found {len(activities)} activities for protocol {protocol_id} (numeric comparison)")
            except Exception as e:
                logging.error(f"Error converting protocol IDs to numeric: {str(e)}")
                # Fallback to original comparison
                activities = self.activities_df[self.activities_df[protocol_id_col] == protocol_id]
            
            if activities.empty:
                logging.warning(f"No activities found for protocol {protocol_id}")
                return pd.DataFrame()
            
            # Process coordinates if available
            if 'Start_coordinates' in activities.columns and 'End_Coordinates' in activities.columns:
                try:
                    # Process coordinates
                    coord_df = self._process_coordinates(activities)
                    # Merge back with activities
                    activities = pd.concat([activities, coord_df], axis=1)
                except Exception as e:
                    logging.error(f"Error processing coordinates: {str(e)}")
                    logging.error(traceback.format_exc())
                    return pd.DataFrame()
            else:
                logging.error(f"Missing coordinate columns in activities data")
                logging.info(f"Available columns: {activities.columns.tolist()}")
                return pd.DataFrame()
                
            # Set up sampling rate for path generation
            self.sampling_rate = 10  # Hz
            
            # Generate path for each activity and combine
            path_points = []
            current_date = datetime.now().date()  # Use current date for datetime objects
            
            for _, activity in activities.iterrows():
                try:
                    # Check for required data
                    if pd.isna(activity['start_lat']) or pd.isna(activity['start_lng']) or \
                    pd.isna(activity['end_lat']) or pd.isna(activity['end_lng']):
                        logging.warning(f"Activity {activity['Activity_ID']} has missing coordinates")
                        continue
                        
                    # Check if times are valid
                    if pd.isna(activity['Start_Time']) or pd.isna(activity['End_Time']):
                        logging.warning(f"Activity {activity['Activity_ID']} has missing times")
                        continue
                        
                    # Generate points for this activity
                    logging.info(f"Generating points for activity {activity['Activity_ID']}")
                    
                    activity_points = self._generate_activity_points(activity, current_date)
                    path_points.extend(activity_points)
                    
                    logging.info(f"Generated {len(activity_points)} points for activity {activity['Activity_ID']}")
                    
                except Exception as e:
                    logging.error(f"Error generating path for activity {activity['Activity_ID']}: {str(e)}")
                    logging.error(traceback.format_exc())
                    continue
            
            if not path_points:
                logging.warning(f"No path points generated for protocol {protocol_id}")
                return pd.DataFrame()
                
            # Create DataFrame from path points
            path_df = pd.DataFrame(path_points)
            
            # Sort by time
            path_df = path_df.sort_values('date time')
            
            logging.info(f"Generated ground truth path with {len(path_df)} points")
            logging.info(f"Time range: {path_df['date time'].min()} to {path_df['date time'].max()}")
            
            return path_df
            
        except Exception as e:
            logging.error(f"Error in get_ground_truth_path: {str(e)}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    def _generate_activity_points(self, activity: pd.Series, base_date: datetime.date) -> List[Dict]:
        try:
            if activity.get('is_circular', False):
                # Handle circular path
                # First ensure times are datetime.time objects
                start_time = activity['Start_Time']
                end_time = activity['End_Time']
                
                # Convert to time objects if they're strings
                if isinstance(start_time, str):
                    try:
                        # Try common time formats
                        for fmt in ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M:%S %p', '%I:%M %p']:
                            try:
                                start_time = datetime.strptime(start_time, fmt).time()
                                break
                            except ValueError:
                                continue
                        else:
                            # If none of the formats worked, use pandas
                            start_time = pd.to_datetime(start_time).time()
                    except Exception as e:
                        logging.error(f"Error converting start time string: {str(e)}")
                        # Default to midnight if conversion fails
                        start_time = time(0, 0, 0)
                
                if isinstance(end_time, str):
                    try:
                        # Try common time formats
                        for fmt in ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M:%S %p', '%I:%M %p']:
                            try:
                                end_time = datetime.strptime(end_time, fmt).time()
                                break
                            except ValueError:
                                continue
                        else:
                            # If none of the formats worked, use pandas
                            end_time = pd.to_datetime(end_time).time()
                    except Exception as e:
                        logging.error(f"Error converting end time string: {str(e)}")
                        # Default to end of day if conversion fails
                        end_time = time(23, 59, 59)
                
                # Create datetime objects using the base_date
                start_dt = datetime.combine(base_date, start_time)
                end_dt = datetime.combine(base_date, end_time)
                
                duration = (end_dt - start_dt).total_seconds()
                steps = int(duration * self.sampling_rate)

                # Use start coordinates as circle center
                if 'first_session' in self.session_id.lower():
                    # For first session, use start point as center
                    center_point = {
                        "lat": activity['start_lat'],
                        "lng": activity['start_lng']
                    }
                else:
                    # For other sessions, use midpoint as center
                    center_point = {
                        "lat": (activity['start_lat'] + activity['end_lat']) / 2,
                        "lng": (activity['start_lng'] + activity['end_lng']) / 2
                    }

                # Transform center point to local coordinates
                center_local = haversine_service.HaversineService.distance_haversine(
                    {"lat": self.pitch_service.get_origin()["lat"], 
                    "lng": self.pitch_service.get_origin()["lng"]},
                    center_point,
                    meters=1000,
                    rotation=self.pitch_service.rotation

                )

                # Fixed 9.9m diameter circle
                if 'first_session' in self.session_id.lower():
                    radius = 4.95  # meters for first_session
                else:
                    radius = 9.0  # meters for other sessions 
                
                # Calculate end point's distance from center if needed
                if not pd.isna(activity['end_lat']) and not pd.isna(activity['end_lng']):
                    end_local = haversine_service.HaversineService.distance_haversine(
                        {"lat": self.pitch_service.get_origin()["lat"],
                        "lng": self.pitch_service.get_origin()["lng"]},
                        {"lat": activity['end_lat'], "lng": activity['end_lng']},
                        meters=1000,
                        rotation=self.pitch_service.rotation

                    )
                    # Optional: Adjust circle to pass through end point while maintaining diameter
                    dx = end_local["dx"] - center_local["dx"]
                    dy = end_local["dy"] - center_local["dy"]
                    end_distance = np.sqrt(dx*dx + dy*dy)
                    if end_distance > 0:
                        # Move center halfway towards end point if end point is further than diameter
                        if end_distance > 9.9:
                            center_local["dx"] += (dx * (end_distance - 9.9)) / (2 * end_distance)
                            center_local["dy"] += (dy * (end_distance - 9.9)) / (2 * end_distance)
                points = []

                for i in range(steps + 1):
                    t = i / steps
                    angle = 2 * np.pi * t
                    
                    # Calculate point on circle in local coordinates
                    x = center_local["dx"] + radius * np.cos(angle)
                    y = center_local["dy"] + radius * np.sin(angle)

                    points.append({
                        'date time': start_dt + timedelta(seconds=duration * t),
                        'x_real_m': x,
                        'y_real_m': y,
                        'Activity_ID': activity['Activity_ID'],
                        'is_circular': True
                    })

                return points
            else:
                # Handle non-circular paths as before
                return self._generate_linear_path(activity, base_date)

        except Exception as e:
            logging.error(f"Error generating activity points: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def _generate_linear_path(self, activity: pd.Series, base_date: datetime.date) -> List[Dict]:
        """Generate points for a linear path between start and end coordinates."""
        try:
            # Handle start time, which might be a string or a time object
            start_time = activity['Start_Time']
            end_time = activity['End_Time']
            
            # Convert to time objects if they're strings
            if isinstance(start_time, str):
                try:
                    # Try common time formats
                    for fmt in ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M:%S %p', '%I:%M %p']:
                        try:
                            start_time = datetime.strptime(start_time, fmt).time()
                            break
                        except ValueError:
                            continue
                    else:
                        # If none of the formats worked, use pandas
                        start_time = pd.to_datetime(start_time).time()
                except Exception as e:
                    logging.error(f"Error converting start time string: {str(e)}")
                    # Default to midnight if conversion fails
                    start_time = time(0, 0, 0)
            
            if isinstance(end_time, str):
                try:
                    # Try common time formats
                    for fmt in ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M:%S %p', '%I:%M %p']:
                        try:
                            end_time = datetime.strptime(end_time, fmt).time()
                            break
                        except ValueError:
                            continue
                    else:
                        # If none of the formats worked, use pandas
                        end_time = pd.to_datetime(end_time).time()
                except Exception as e:
                    logging.error(f"Error converting end time string: {str(e)}")
                    # Default to end of day if conversion fails
                    end_time = time(23, 59, 59)
            
            # Create datetime objects using the base_date
            start_dt = datetime.combine(base_date, start_time)
            end_dt = datetime.combine(base_date, end_time)
            
            duration = (end_dt - start_dt).total_seconds()
            steps = int(duration * self.sampling_rate)
            
            start_point = {"lat": activity['start_lat'], "lng": activity['start_lng']}
            end_point = {"lat": activity['end_lat'], "lng": activity['end_lng']}
            
            points = []
            origin = {"lat": self.pitch_service.get_origin()["lat"], 
                    "lng": self.pitch_service.get_origin()["lng"]}

            for i in range(steps + 1):
                t = i / steps
                # Interpolate between start and end points
                current_lat = start_point["lat"] + (end_point["lat"] - start_point["lat"]) * t
                current_lng = start_point["lng"] + (end_point["lng"] - start_point["lng"]) * t
                
                # Transform to local coordinates
                current_local = haversine_service.HaversineService.distance_haversine(
                    origin,
                    {"lat": current_lat, "lng": current_lng},
                    meters=1000,
                    rotation=self.pitch_service.rotation

                )

                points.append({
                    'date time': start_dt + timedelta(seconds=duration * t),
                    'x_real_m': current_local["dx"],
                    'y_real_m': current_local["dy"],
                    'Activity_ID': activity['Activity_ID'],
                    'is_circular': False
                })

            return points

        except Exception as e:
            logging.error(f"Error generating linear path: {str(e)}")
            logging.error(traceback.format_exc())
            return []
    def get_protocol_summary(self, protocol_id: int) -> Dict:
        """
        Get summary statistics for a protocol.
        
        Args:
            protocol_id: ID of the protocol
            
        Returns:
            Dictionary containing protocol summary
        """
        try:
            protocol = self.protocols_df[
                self.protocols_df['Protocol ID'] == protocol_id
            ].iloc[0]
            
            activities = self.activities_df[
                self.activities_df['Protocol_ID'] == protocol_id
            ]
            
            summary = {
                'protocol_name': protocol['Protocol name'],
                'start_time': protocol['Start_Time'].strftime('%H:%M:%S'),
                'end_time': protocol['End_Time'].strftime('%H:%M:%S'),
                'multi_analysis': bool(protocol['multi_analysis']),
                'total_activities': len(activities),
                'total_distance': self._calculate_total_distance(activities)
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating protocol summary: {str(e)}")
            return {}

    def _calculate_total_distance(self, activities: pd.DataFrame) -> float:
        """
        Calculate total distance covered in activities.
        
        Args:
            activities: DataFrame of activities
            
        Returns:
            Total distance in meters
        """
        total_distance = 0.0
        
        for _, activity in activities.iterrows():
            # Calculate distance between start and end points
            lat1, lng1 = activity['start_lat'], activity['start_lng']
            lat2, lng2 = activity['end_lat'], activity['end_lng']
            
            # Simple distance calculation (can be enhanced with Haversine)
            dx = (lat2 - lat1) * 111320  # Approximate meters per degree at equator
            dy = (lng2 - lng1) * 111320 * np.cos(np.radians((lat1 + lat2) / 2))
            
            distance = np.sqrt(dx**2 + dy**2)
            total_distance += distance
            
        return total_distance
    
    def _validate_activity_transition(self, last_point: Dict, first_point: Dict) -> bool:
        """Validate the transition between two activities."""
        try:
            time_diff = (pd.to_datetime(first_point['date time']) - 
                        pd.to_datetime(last_point['date time'])).total_seconds()
            
            coord_diff = np.sqrt(
                (first_point['lat'] - last_point['lat'])**2 +
                (first_point['lng'] - last_point['lng'])**2
            )
            
            return time_diff < 1.0 and coord_diff < 1e-6  # 1 second and small coordinate difference
            
        except Exception as e:
            logging.error(f"Error validating activity transition: {str(e)}")
            return False