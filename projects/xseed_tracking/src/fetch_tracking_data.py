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
DATA_DIR = PROJECT_ROOT / "data" / "Data_to_be_fetched"/"club milano data"  # Put your CSV file here
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "club_moilano_fetched_data"

# # API configuration
# API configuration
API_HOSTNAME = "xseed-advanced-metrics-btb-env.eba-p2hm3mer.eu-south-1.elasticbeanstalk.com"
API_ENDPOINT = f"http://{API_HOSTNAME}/trace"  # No port specified
TIMEOUT = 90

# Use the IP address with the port for the endpoint
logger.info(f"Using API endpoint: {API_ENDPOINT} with hostname: {API_HOSTNAME}")
TIMEOUT = 90
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
    
    # Set default headers - no Host header needed when using hostname directly
    session.headers.update({
        'Accept': 'application/json',
        'Content-Type': 'application/json'
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
        
        # Make the request directly without using a session
        response = requests.get(API_ENDPOINT, params=params, timeout=TIMEOUT)
        
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
    user_id = int(float(row['user_id'])) if not pd.isna(row['user_id']) else None
    session_id = int(float(row['session_id'])) if not pd.isna(row['session_id']) else None
    segment_id = int(float(row['id'])) if not pd.isna(row['id']) else None  # Using 'id' as segment_id
    
    if None in [user_id, session_id, segment_id]:
        logger.warning(f"Skipping row with missing data: {row}")
        return None
    
    output_file = Path(output_dir) / f"player_{user_id}_session_{session_id}_segment_{segment_id}.csv"
    
    df = get_trace_data(
        user_id=user_id,
        session_id=session_id,
        segment_id=segment_id,
        trace_type=trace_type,
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
        
        # Validate required columns
        required_columns = ['user_id', 'session_id', 'id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in CSV: {missing_columns}")
            return {}
        
        # Drop rows with missing values in required columns
        initial_rows = len(df)
        df = df.dropna(subset=required_columns)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing values")
        
        # Get unique combinations
        unique_combinations = df.drop_duplicates(subset=['user_id', 'session_id', 'id'])
        logger.info(f"Found {len(unique_combinations)} unique player/session/segment combinations")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process in parallel
        results = {}
        failed_combinations = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for each unique combination
            futures = {
                executor.submit(
                    fetch_player_data,
                    row=row,
                    output_dir=output_dir,
                    trace_type=trace_type,
                    version=version
                ): (row['user_id'], row['session_id'], row['id'])
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
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing input file: {e}")
        return {}

def main():
    """Main execution function."""
    logger.info("Starting tracking data fetch")
    
    # Configure paths
    input_csv = DATA_DIR / "1236_metadata.csv"
    if not input_csv.exists():
        logger.error(f"Input file not found: {input_csv}")
        return 1
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    trace_type = "match"
    version = 1
    max_workers = 3  # Limit to avoid overwhelming the server
    
    logger.info(f"Input file: {input_csv}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Trace type: {trace_type}")
    logger.info(f"API version: {version}")
    logger.info(f"Max parallel workers: {max_workers}")
    logger.info(f"API endpoint: {API_ENDPOINT}")
    
    # Fetch data for all players
    player_data = fetch_all_players_data(
        input_csv=str(input_csv),
        output_dir=str(OUTPUT_DIR),
        trace_type=trace_type,
        version=version,
        max_workers=max_workers
    )
    
    # Create combined dataset if any data was fetched
    if player_data:
        logger.info(f"Successfully fetched data for {len(player_data)} players")
        
        try:
            combined_output = OUTPUT_DIR / "combined_trace_data.csv"
            combined_df = pd.concat(
                [df.assign(user_id=user_id) for user_id, df in player_data.items()],
                ignore_index=True
            )
            combined_df.to_csv(combined_output, index=False)
            logger.info(f"Combined dataset saved to {combined_output}")
        except Exception as e:
            logger.error(f"Error creating combined dataset: {e}")
    else:
        logger.warning("No data was successfully fetched")
    
    logger.info("Data fetch completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())