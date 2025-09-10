# run_debug.py - Standalone debugging script for player tracking system
# Run this with: python run_debug.py

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_output.log'),
        logging.StreamHandler()
    ]
)

def debug_data_files(session_id):
    """
    Diagnose issues with protocol and activity files.
    
    Args:
        session_id: ID of the session to debug
    """
    logging.info(f"\n==== DEBUGGING SESSION: {session_id} ====\n")
    base_path = Path(f'data/{session_id}')
    
    # Summarize available files
    logging.info(f"Files in {base_path}:")
    try:
        all_files = list(base_path.glob('*'))
        for file in all_files:
            logging.info(f"  - {file.name} ({file.stat().st_size} bytes)")
    except Exception as e:
        logging.error(f"Error listing files: {str(e)}")
    
    # Debug protocol files
    for i in range(1, 4):
        for ext in ['.csv', '.xlsx', '.xls']:
            protocol_file = base_path / f'Protocol_Sheet_{i}{ext}'
            if protocol_file.exists():
                logging.info(f"\n----- Debugging Protocol_Sheet_{i}{ext} -----")
                
                # Try to read raw content
                try:
                    with open(protocol_file, 'rb') as f:
                        content = f.read(200)  # Read first 200 bytes
                        logging.info(f"File starts with: {content}")
                except Exception as e:
                    logging.error(f"Error reading file: {str(e)}")
                
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(protocol_file, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:
                                logging.info(f"Successfully read with encoding={encoding}, sep={sep}")
                                logging.info(f"Columns: {df.columns.tolist()}")
                                logging.info(f"First row: {df.iloc[0].tolist() if len(df) > 0 else 'Empty'}")
                                break
                        except Exception as e:
                            pass
                
                # Try Excel format if it's an Excel file
                if ext in ['.xlsx', '.xls']:
                    try:
                        df = pd.read_excel(protocol_file)
                        logging.info(f"Successfully read as Excel")
                        logging.info(f"Columns: {df.columns.tolist()}")
                        logging.info(f"First row: {df.iloc[0].tolist() if len(df) > 0 else 'Empty'}")
                    except Exception as e:
                        logging.error(f"Error reading as Excel: {str(e)}")
    
    # Debug activity files
    for i in range(1, 4):
        for ext in ['.csv', '.xlsx', '.xls']:
            activity_file = base_path / f'Activity_Sheet_{i}{ext}'
            if activity_file.exists():
                logging.info(f"\n----- Debugging Activity_Sheet_{i}{ext} -----")
                
                # Try to read raw content
                try:
                    with open(activity_file, 'rb') as f:
                        content = f.read(200)  # Read first 200 bytes
                        logging.info(f"File starts with: {content}")
                except Exception as e:
                    logging.error(f"Error reading file: {str(e)}")
                
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(activity_file, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:
                                logging.info(f"Successfully read with encoding={encoding}, sep={sep}")
                                logging.info(f"Columns: {df.columns.tolist()}")
                                logging.info(f"First row: {df.iloc[0].tolist() if len(df) > 0 else 'Empty'}")
                                break
                        except Exception as e:
                            pass
                
                # Try Excel format if it's an Excel file
                if ext in ['.xlsx', '.xls']:
                    try:
                        df = pd.read_excel(activity_file)
                        logging.info(f"Successfully read as Excel")
                        logging.info(f"Columns: {df.columns.tolist()}")
                        logging.info(f"First row: {df.iloc[0].tolist() if len(df) > 0 else 'Empty'}")
                    except Exception as e:
                        logging.error(f"Error reading as Excel: {str(e)}")
    
    # Also check stadium properties
    stadium_file = base_path / 'stadium_properties.csv'
    if stadium_file.exists():
        logging.info(f"\n----- Debugging stadium_properties.csv -----")
        
        # Try to read raw content
        try:
            with open(stadium_file, 'rb') as f:
                content = f.read(200)  # Read first 200 bytes
                logging.info(f"File starts with: {content}")
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
        
        # Try different encodings and separators
        for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(stadium_file, encoding=encoding, sep=sep)
                    if len(df.columns) > 1:
                        logging.info(f"Successfully read with encoding={encoding}, sep={sep}")
                        logging.info(f"Columns: {df.columns.tolist()}")
                        logging.info(f"First row: {df.iloc[0].tolist() if len(df) > 0 else 'Empty'}")
                        break
                except Exception as e:
                    pass

def debug_ground_truth_processor(session_id, player_num=1):
    """
    Debug the GroundTruthProcessor initialization and ground truth generation
    
    Args:
        session_id: ID of the session to debug
        player_num: Player number (1, 2, or 3)
    """
    logging.info(f"\n==== DEBUGGING GROUND TRUTH PROCESSOR: {session_id}, Player {player_num} ====\n")
    
    try:
        # Import required modules - may need to adjust these imports based on your project structure
        try:
            from src.ground_truth_processor import GroundTruthProcessor
            from services.pitch_service import PitchService
        except ImportError:
            # Try relative imports if the above fails
            try:
                sys.path.append('.')  # Add current directory to path
                from ground_truth_processor import GroundTruthProcessor
                from pitch_service import PitchService
            except ImportError:
                logging.error("Could not import required modules. Please check your Python path.")
                return
        
        # Initialize PitchService
        stadium_path = f'data/{session_id}/stadium_properties.csv'
        logging.info(f"Initializing PitchService with {stadium_path}")
        
        try:
            pitch_service = PitchService(stadium_path)
            logging.info("PitchService initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing PitchService: {str(e)}")
            logging.error(traceback.format_exc())
            return
        
        # Initialize GroundTruthProcessor
        protocol_file = f'data/{session_id}/Protocol_Sheet_{player_num}.csv'
        activity_file = f'data/{session_id}/Activity_Sheet_{player_num}.csv'
        
        logging.info(f"Initializing GroundTruthProcessor with:")
        logging.info(f"  - Protocol file: {protocol_file}")
        logging.info(f"  - Activity file: {activity_file}")
        
        try:
            gt_processor = GroundTruthProcessor(protocol_file, [activity_file], pitch_service)
            logging.info("GroundTruthProcessor initialized successfully")
            
            # Log loaded data
            logging.info(f"Protocols loaded: {len(gt_processor.protocols_df)} rows")
            if not gt_processor.protocols_df.empty:
                logging.info(f"Protocol columns: {gt_processor.protocols_df.columns.tolist()}")
                logging.info(f"Protocol sample: {gt_processor.protocols_df.iloc[0].to_dict()}")
            
            logging.info(f"Activities loaded: {len(gt_processor.activities_df)} rows")
            if not gt_processor.activities_df.empty:
                logging.info(f"Activity columns: {gt_processor.activities_df.columns.tolist()}")
                logging.info(f"Activity sample: {gt_processor.activities_df.iloc[0].to_dict()}")
            
            # Try to generate ground truth for each protocol
            if not gt_processor.protocols_df.empty and 'Protocol ID' in gt_processor.protocols_df.columns:
                for protocol_id in gt_processor.protocols_df['Protocol ID'].unique():
                    logging.info(f"\nGetting ground truth for protocol {protocol_id}")
                    try:
                        ground_truth = gt_processor.get_ground_truth_path(protocol_id)
                        if ground_truth is not None and not ground_truth.empty:
                            logging.info(f"Successfully generated ground truth with {len(ground_truth)} points")
                            logging.info(f"Ground truth columns: {ground_truth.columns.tolist()}")
                            logging.info(f"Ground truth sample: {ground_truth.iloc[0].to_dict()}")
                        else:
                            logging.warning(f"Ground truth is empty for protocol {protocol_id}")
                    except Exception as e:
                        logging.error(f"Error generating ground truth for protocol {protocol_id}: {str(e)}")
                        logging.error(traceback.format_exc())
            
        except Exception as e:
            logging.error(f"Error initializing GroundTruthProcessor: {str(e)}")
            logging.error(traceback.format_exc())
    
    except Exception as e:
        logging.error(f"Error in debug_ground_truth_processor: {str(e)}")
        logging.error(traceback.format_exc())

# Run the debug functions
if __name__ == "__main__":
    # Get session ID from command line argument or use default
    session_id = sys.argv[1] if len(sys.argv) > 1 else 'first_session'
    player_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    logging.info(f"Running debug for session '{session_id}', player {player_num}")
    
    # Debug data files
    debug_data_files(session_id)
    
    # Debug ground truth processor
    debug_ground_truth_processor(session_id, player_num)
    
    logging.info("Debug complete. Check debug_output.log for full details.")
    
    # Print instructions for using the script
    print("\nUsage instructions:")
    print("  python run_debug.py [session_id] [player_num]")
    print("  - session_id: The session to debug (default: first_session)")
    print("  - player_num: The player number to debug (default: 1)")
    print("\nExample: python run_debug.py first_session 2")
    print("\nResults are saved to debug_output.log")