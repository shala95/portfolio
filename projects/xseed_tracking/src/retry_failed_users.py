import pandas as pd
import requests
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define paths
OUTPUT_DIR = Path("/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/outputs/fetched_data")

# API configuration
API_IP = "15.161.76.156"
API_HOSTNAME = "xseed-advanced-metrics-btb-env.eba-p2hm3mer.eu-south-1.elasticbeanstalk.com"
API_ENDPOINT = f"http://{API_IP}/trace"

# Failed combinations to retry5505
FAILED_COMBINATIONS = [
    {"user_id": 9431, "session_id": 30888, "segment_id": 5505, "type": "match"},
    {"user_id": 9432, "session_id": 30889, "segment_id": 5506, "type": "match"},
    {"user_id": 9436, "session_id": 30893, "segment_id": 5507, "type": "match"}
]

def get_trace_data_with_retry(user_id, session_id, segment_id, trace_type="match", 
                             max_retries=5, backoff_factor=5, output_file=None):
    """
    Fetch data with exponential backoff retry strategy and extended timeouts.
    This function tries multiple approaches if the first one fails.
    """
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Host': API_HOSTNAME
    }
    
    params = {
        "user_id": user_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "type": trace_type,
        "version": 1
    }
    
    # First attempt with direct IP
    logger.info(f"Attempting to fetch data for user={user_id}, session={session_id}, segment={segment_id}")
    
    for attempt in range(max_retries):
        try:
            # Increase timeout for potentially slow responses
            response = requests.get(
                API_ENDPOINT, 
                params=params, 
                headers=headers, 
                timeout=180
            )
            
            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data")
                
                if data:
                    df = pd.DataFrame(data)
                    logger.info(f"Successfully retrieved {len(df)} records")
                    
                    # Save to file if requested
                    if output_file:
                        output_path = Path(output_file)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(output_file, index=False)
                        logger.info(f"Data saved to {output_file}")
                    
                    return df
                else:
                    logger.warning(f"Empty data array returned on attempt {attempt+1}")
            else:
                logger.warning(f"Received status code {response.status_code} on attempt {attempt+1}")
                
            # If we're here, we either got a non-200 status code or empty data
            if attempt < max_retries - 1:
                # Calculate exponential backoff wait time
                wait_time = backoff_factor * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts")
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts")
    
    # If we've exhausted all retries, try a different approach (hostname instead of IP)
    logger.info("Trying alternative approach with hostname...")
    try:
        alt_endpoint = f"http://{API_HOSTNAME}/trace"
        response = requests.get(alt_endpoint, params=params, timeout=180)
        
        if response.status_code == 200:
            response_json = response.json()
            data = response_json.get("data")
            
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Alternative approach succeeded with {len(df)} records")
                
                if output_file:
                    output_path = Path(output_file)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_file, index=False)
                    logger.info(f"Data saved to {output_file}")
                
                return df
    except Exception as e:
        logger.error(f"Alternative approach failed: {e}")
    
    return pd.DataFrame()

def update_combined_data(new_file_path, combined_file_path):
    """
    Update the combined dataset with new data
    """
    try:
        # Read the new data file
        new_data = pd.read_csv(new_file_path)
        
        # Check if combined file exists
        combined_file = Path(combined_file_path)
        if combined_file.exists():
            # Read the existing combined file
            combined_data = pd.read_csv(combined_file)
            
            # Determine user_id from the filename if not present in data
            if 'user_id' not in new_data.columns:
                user_id = int(new_file_path.split("player_")[1].split("_session_")[0])
                new_data['user_id'] = user_id
            
            # Append the new data
            updated_data = pd.concat([combined_data, new_data], ignore_index=True)
            
            # Save the updated file
            updated_data.to_csv(combined_file, index=False)
            logger.info(f"Updated combined dataset with {len(new_data)} new records. New total: {len(updated_data)}")
        else:
            logger.warning(f"Combined file not found at {combined_file}")
    except Exception as e:
        logger.error(f"Error updating combined data: {e}")

def main():
    logger.info("Starting retry process for failed user IDs")
    
    combined_file = OUTPUT_DIR / "combined_trace_data.csv"
    success_count = 0
    
    for combo in FAILED_COMBINATIONS:
        user_id = combo["user_id"]
        session_id = combo["session_id"]
        segment_id = combo["segment_id"]
        trace_type = combo["type"]
        
        output_file = OUTPUT_DIR / f"player_{user_id}_session_{session_id}_segment_{segment_id}.csv"
        
        logger.info(f"Processing: user={user_id}, session={session_id}, segment={segment_id}")
        
        # Add extra delay between requests to avoid overwhelming server
        time.sleep(10)
        
        df = get_trace_data_with_retry(
            user_id=user_id,
            session_id=session_id,
            segment_id=segment_id,
            trace_type=trace_type,
            max_retries=5,
            backoff_factor=10,  # 10, 20, 40, 80, 160 seconds
            output_file=str(output_file)
        )
        
        if not df.empty:
            success_count += 1
            update_combined_data(output_file, combined_file)
    
    logger.info(f"Retry process completed. Successfully retrieved {success_count}/{len(FAILED_COMBINATIONS)} combinations")

if __name__ == "__main__":
    main()