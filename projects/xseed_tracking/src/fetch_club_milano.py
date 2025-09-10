import pandas as pd
from pathlib import Path
import requests
import time
import concurrent.futures
import logging
import os
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tracking_data_fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define project paths - modify these according to your project structure
PROJECT_ROOT = Path("/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking")
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "Data_to_be_fetched"/"club milano data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "club_moilano_fetched_data"

# API configuration - Using direct IP address with hostname in header
API_IP = "15.161.76.156"
API_HOSTNAME = "xseed-advanced-metrics-btb-env.eba-p2hm3mer.eu-south-1.elasticbeanstalk.com"
API_ENDPOINT = f"http://{API_IP}/trace"
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 1
RETRY_STATUS_CODES = [500, 502, 503, 504]

def create_session():
    """Create a requests session with retry logic and proper headers."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS_CODES
    )
    
    # Mount the adapter to the session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default headers
    session.headers.update({
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Host': API_HOSTNAME  # Include original hostname for virtual hosting
    })
    
    return session

def get_trace_data(user_id, session_id, segment_id, trace_type="match", version=1, output_file=None):
    """
    Fetch tracking data for a specific player/session/segment combination.
    
    Args:
        user_id (int): The user ID
        session_id (int): The session ID
        segment_id (int): The segment ID (using 'id' field from CSV)
        trace_type (str): Type of trace (default: "match")
        version (int): API version (default: 1)
        output_file (str): Path to save the output file (optional)
        
    Returns:
        DataFrame: Fetched tracking data
    """
    # Ensure all IDs are integers
    user_id = int(float(user_id)) if user_id is not None else None
    session_id = int(float(session_id)) if session_id is not None else None
    segment_id = int(float(segment_id)) if segment_id is not None else None
    
    # Ensure trace_type is lowercase
    trace_type = str(trace_type).lower().strip()
    
    # Create session with retry logic
    session = create_session()
    
    # Create request parameters
    params = {
        "user_id": user_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "type": trace_type,
        "version": version
    }
    
    try:
        logger.info(f"Fetching data for user={user_id}, session={session_id}, segment={segment_id}")
        
        # Make the request
        response = session.get(API_ENDPOINT, params=params, timeout=TIMEOUT)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        response_json = response.json()
        
        # Check for data
        data = response_json.get("data")
        if not data:
            logger.warning(f"No data returned for user={user_id}, session={session_id}, segment={segment_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to file if requested
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_file, index=False)
                logger.info(f"Data saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving data to {output_file}: {e}")
        
        logger.info(f"Successfully fetched {len(df)} records for user={user_id}")
        return df
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 500:
            logger.error(f"Server error for user={user_id}: {e}")
        else:
            logger.error(f"HTTP error for user={user_id}: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for user={user_id}: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error for user={user_id}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for user={user_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for user={user_id}: {e}")
    
    return pd.DataFrame()

def fetch_player_data(row, output_dir, trace_type="match", version=1):
    """Process a single player row, for use with parallel processing."""
    # Extract user_id, session_id, and segment_id
    user_id = row.get('user_id') if not pd.isna(row.get('user_id')) else None
    session_id = row.get('session_id') if not pd.isna(row.get('session_id')) else None
    
    # Use 'segment_id' if available, otherwise use 'id' as segment_id
    segment_id = None
    if 'segment_id' in row and not pd.isna(row['segment_id']):
        segment_id = row['segment_id']
    elif 'id' in row and not pd.isna(row['id']):
        segment_id = row['id']
    
    # Ensure all IDs are integers
    user_id = int(float(user_id)) if user_id is not None else None
    session_id = int(float(session_id)) if session_id is not None else None
    segment_id = int(float(segment_id)) if segment_id is not None else None
    
    if None in [user_id, session_id, segment_id]:
        logger.warning(f"Skipping row with missing data: {row.to_dict()}")
        return None
    
    output_file = Path(output_dir) / f"player_{user_id}_session_{session_id}_segment_{segment_id}.csv"
    
    df = get_trace_data(
        user_id=user_id,
        session_id=session_id,
        segment_id=segment_id,
        trace_type=trace_type.lower(),  # Ensure trace_type is lowercase
        version=version,
        output_file=str(output_file)
    )
    
    if not df.empty:
        return (user_id, df)
    return None

def fetch_all_players_data(input_csv, output_dir, trace_type="match", version=1, max_workers=5):
    """
    Fetch tracking data for all players in the input CSV file using parallel processing.
    
    Args:
        input_csv (str): Path to input CSV file
        output_dir (str): Directory to save output files
        trace_type (str): Type of trace (default: "match")
        version (int): API version (default: 1)
        max_workers (int): Maximum number of parallel workers (default: 5)
        
    Returns:
        dict: Dictionary mapping user_ids to DataFrames
    """
    logger.info(f"Reading input file: {input_csv}")
    
    try:
        # Read CSV file
        df = pd.read_csv(input_csv)
        
        # Determine which columns to use for required data
        # This makes the code flexible for different CSV formats
        required_data = ['user_id', 'session_id']
        segment_columns = ['segment_id', 'id']
        
        # Check if we have the minimum required columns
        missing_columns = [col for col in required_data if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns in CSV: {missing_columns}")
            return {}
        
        # Check if we have at least one segment identifier column
        segment_available = any(col in df.columns for col in segment_columns)
        if not segment_available:
            logger.error(f"Missing segment identifier columns (one of {segment_columns} is required)")
            return {}
        
        # Map a segment column to 'segment_id' if needed
        # Always use 'id' column as the segment identifier if it exists
        if 'id' in df.columns:
            logger.info("Using 'id' column as segment identifier for all rows")
            df['segment_id'] = df['id']
            # If we already have a segment_id column, we're overwriting it with id values
            if 'segment_id' in df.columns:
                logger.info("Overwriting existing 'segment_id' column with values from 'id' column")
        else:
            # If no 'id' column exists but we have segment_id, leave it as is
            if 'segment_id' in df.columns:
                logger.info("Using existing 'segment_id' column (no 'id' column found)")
            else:
                # Neither column exists
                logger.error("Missing both 'id' and 'segment_id' columns in CSV")
                return {}
        
        # Drop rows with missing values in required columns and segment column
        required_for_drop = required_data + ['segment_id'] if 'segment_id' in df.columns else required_data + ['id']
        initial_rows = len(df)
        df = df.dropna(subset=required_for_drop)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        # Get unique combinations
        if 'segment_id' in df.columns:
            unique_combinations = df.drop_duplicates(subset=['user_id', 'session_id', 'segment_id'])
        else:
            unique_combinations = df.drop_duplicates(subset=['user_id', 'session_id', 'id'])
            
        logger.info(f"Found {len(unique_combinations)} unique player/session/segment combinations")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine output directory name based on input file
        input_file_name = Path(input_csv).stem
        specific_output_dir = Path(output_dir) / input_file_name
        specific_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving data to {specific_output_dir}")
        
        # Process in parallel
        results = {}
        failed_combinations = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for each unique combination
            futures = {
                executor.submit(
                    fetch_player_data,
                    row=row,
                    output_dir=specific_output_dir,
                    trace_type=trace_type,
                    version=version
                ): (row['user_id'], row['session_id'], row.get('segment_id', row.get('id')))
                for _, row in unique_combinations.iterrows()
            }
            
            # Process as they complete
            for future in concurrent.futures.as_completed(futures):
                user_id, session_id, segment_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        user_id, player_df = result
                        results[user_id] = player_df
                except Exception as e:
                    logger.error(f"Error processing user={user_id}, session={session_id}, segment={segment_id}: {e}")
                    failed_combinations.append((user_id, session_id, segment_id))
        
        # Print summary of failed combinations
        if failed_combinations:
            logger.warning("Failed to fetch data for the following combinations:")
            for user_id, session_id, segment_id in failed_combinations:
                logger.warning(f"  user={user_id}, session={session_id}, segment={segment_id}")
        
        # Create combined dataset if any data was fetched
        if results:
            logger.info(f"Successfully fetched data for {len(results)} players")
            
            try:
                combined_output = specific_output_dir / f"{input_file_name}_combined_trace_data.csv"
                combined_df = pd.concat(
                    [df.assign(user_id=user_id) for user_id, df in results.items()],
                    ignore_index=True
                )
                combined_df.to_csv(combined_output, index=False)
                logger.info(f"Combined dataset saved to {combined_output}")
            except Exception as e:
                logger.error(f"Error creating combined dataset: {e}")
        else:
            logger.warning("No data was successfully fetched")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing input file: {e}")
        return {}

def main():
    """Main execution function."""
    logger.info("Starting tracking data fetch")
    
    # Configure paths
    input_files = [
        DATA_DIR / "1223_metadata.csv",
        DATA_DIR / "1236_metadata.csv",
        DATA_DIR / "698_data_for_MS.csv"
    ]
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    version = 1
    max_workers = 1  # REDUCED to avoid overwhelming the server
    
    # Increase the global timeout
    global TIMEOUT
    TIMEOUT = 120  # Increased from 30 seconds to 120 seconds
    
    for input_csv in input_files:
        if not input_csv.exists():
            logger.error(f"Input file not found: {input_csv}")
            continue
        
        # Determine appropriate trace_type based on file content
        try:
            df = pd.read_csv(input_csv)
            # Check if the 'type' column exists
            if 'type' in df.columns and not df['type'].empty:
                # Get the first non-null value and convert to lowercase
                first_type = df['type'].dropna().iloc[0] if not df['type'].dropna().empty else "match"
                trace_type = str(first_type).lower().strip()
                logger.info(f"Detected trace_type from file: {first_type} (using as: {trace_type})")
            else:
                trace_type = "match"
                logger.info(f"No type column or values found, using default trace_type: {trace_type}")
        except Exception as e:
            trace_type = "match"
            logger.warning(f"Error detecting trace_type from file, using default: {trace_type}. Error: {e}")
        
        # Determine output directory name based on input file
        input_file_name = Path(input_csv).stem
        specific_output_dir = OUTPUT_DIR / input_file_name
        specific_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n\n===== Processing file: {input_csv} =====")
        logger.info(f"Output directory: {specific_output_dir}")
        logger.info(f"Trace type: {trace_type}")
        logger.info(f"API version: {version}")
        logger.info(f"Max parallel workers: {max_workers}")
        logger.info(f"Request timeout: {TIMEOUT} seconds")
        
        # Add a small delay before starting to fetch data
        logger.info("Waiting 5 seconds before starting requests...")
        time.sleep(5)
        
        # Fetch data for all players in this file
        player_data = fetch_all_players_data(
            input_csv=str(input_csv),
            output_dir=str(specific_output_dir),
            trace_type=trace_type,
            version=version,
            max_workers=max_workers
        )
        
        # Add a small delay between files
        if input_csv != input_files[-1]:  # If not the last file
            logger.info("Waiting 30 seconds before processing next file...")
            time.sleep(30)
    
    logger.info("All data fetch operations completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())