import requests
import pandas as pd
from pathlib import Path

def get_trace_data(user_id: int, session_id: int, segment_id: int, trace_type: str, 
                   output_file: str = None, version: int = 1) -> pd.DataFrame:
    """
    Fetch data from the given endpoint, extract the "data" field, return it as a DataFrame,
    and optionally save it to a CSV file.

    Args:
        user_id (int): User ID parameter.
        session_id (int): Session ID parameter.
        segment_id (int): Segment ID parameter.
        trace_type (str): Type of trace.
        output_file (str, optional): Path to save the CSV file. If None, data won't be saved.
        version (int, optional): Version number. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame created from the "data" field of the response JSON.
    """
    # Endpoint URL
    url = "http://xseed-advanced-metrics-btb-env.eba-p2hm3mer.eu-south-1.elasticbeanstalk.com/trace"

    # Parameters
    params = {
        "user_id": user_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "type": trace_type,
        "version": version
    }

    try:
        # Send GET request
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse response JSON
        response_json = response.json()
        
        # Extract "data" field
        data = response_json.get("data")
        if not data:
            raise ValueError("No 'data' field found in the response JSON.")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV if output_file is specified
        if output_file:
            try:
                # Create directory if it doesn't exist
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Data successfully saved to {output_file}")
            except Exception as e:
                print(f"Error saving CSV file: {e}")
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return pd.DataFrame()  # Return an empty DataFrame on failure

# Example usage
if __name__ == "__main__":
    # Test parameters
    user_id = 9333
    session_id = 31205
    segment_id = 6840
    trace_type = "training"
    version = 1
    
    # Specify output file path
    output_file = "/Users/mohamedshoala/Desktop/trace_data.csv"  # You can change this to include a directory path

    # Call function
    df = get_trace_data(user_id, session_id, segment_id, trace_type, output_file, version)
    print(df)