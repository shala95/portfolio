from src.data_processor import DataProcessor
from src.display_manager import DisplayManager
from src.comparison_processor import ComparisonProcessor
from simple_hardware_comparison import run_simple_hardware_comparison, calculate_hardware_comparison, display_hardware_comparison
import os
import sys
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import traceback
import logging
import traceback
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gps_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_available_sessions() -> List[str]:
    base_path = Path('data')
    return [d.name for d in base_path.iterdir() 
            if d.is_dir() and d.name != 'output']



def display_session_info(session_id: str):
    """Display information about the current session."""
    st.sidebar.markdown("### Session Information")
    
    # Create a DataProcessor to get the actual player list
    try:
        temp_processor = DataProcessor(session_id)
        player_list = temp_processor.get_players_list()
        player_count = len(player_list)
        player_numbers = [p.split()[-1] for p in player_list]
    except:
        # Fallback if DataProcessor can't be created
        player_count = 0
        player_numbers = []
        player_list = []
    
    # Check which types of files exist
    base_path = Path(f'data/{session_id}')
    
    # Detect file formats
    formats = detect_file_formats(base_path)
    
    # Check for Xseed versions
    xseed_old = any(check_file_exists(base_path / f'player_{p.replace(".", "_")}/df_trace_old', True) 
                   for p in player_numbers)
    xseed_new = any(check_file_exists(base_path / f'player_{p.replace(".", "_")}/df_trace_new', True) 
                   for p in player_numbers)
    xseed_standard = any(check_file_exists(base_path / f'player_{p.replace(".", "_")}/df_trace', True) 
                        for p in player_numbers)
    
    # Display info
    st.sidebar.markdown(f"**Session ID:** {session_id}")
    st.sidebar.markdown(f"**Players:** {player_count} ({', '.join(player_list)})")
    st.sidebar.markdown(f"**File Formats:** {formats['csv']} CSV, {formats['xlsx']+formats['xls']} Excel")
    
    gpexe_exists = any(check_file_exists(base_path / f'gpexe_track_{p}', True) for p in player_numbers)
    st.sidebar.markdown(f"**GPEXE Data:** {'✅' if gpexe_exists else '❌'}")
    st.sidebar.markdown(f"**Xseed Old HW:** {'✅' if xseed_old else '❌'}")
    st.sidebar.markdown(f"**Xseed New HW:** {'✅' if xseed_new else '❌'}")
    if xseed_standard:
        st.sidebar.markdown(f"**Xseed Standard:** ✅")

def ensure_session_directories(session_id: str):
    """Create required directories for a specific session."""
    # Base directories
    required_dirs = [
        f'data/{session_id}',
        'output'
    ]
    
    # Check for player directories based on existing files
    base_path = Path(f'data/{session_id}')
    player_numbers = []
    
    # Helper function to check file existence
    def check_any_format(prefix, num):
        """Check if file exists in any format."""
        for ext in ['.csv', '.xlsx', '.xls']:
            if (base_path / f'{prefix}_{num}{ext}').exists():
                return True
        return False
    
    # Check for standard players
    for i in range(1, 7):
        if any(check_any_format(prefix, i) 
               for prefix in ['gpexe_track', 'Activity_Sheet', 'Protocol_Sheet']):
            player_numbers.append(str(i))
    
    
    if not player_numbers:
        player_numbers = ['1', '2', '3']  # Default to 3 if none found
    
    # Add player directories with decimal handling
    for player_num in player_numbers:
        required_dirs.append(f'data/{session_id}/player_{player_num.replace(".", "_")}')
    
    # Create all directories
    for dir_name in required_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

def check_session_files(session_id: str, strict=False):
    """
    Check if required files exist for the session.
    With strict=False, will only warn about missing files instead of raising errors.
    """
    base_path = Path(f'data/{session_id}')
    
    # Create a DataProcessor to get the actual player list
    temp_processor = DataProcessor(session_id)
    player_list = temp_processor.get_playsers_list()
    
    # Check essential files
    missing_files = []
    
    # Stadium properties is essential
    if not any((base_path / f'stadium_properties{ext}').exists() 
               for ext in ['.csv', '.xlsx', '.xls']):
        missing_files.append(f"Stadium properties")
    
    # Check file sets for each detected player
    warnings = []
    for player in player_list:
        player_num = player.split()[-1]
        
        # Check GPS track file
        if not check_file_exists(base_path / f'gpexe_track_{player_num}', True):
            warnings.append(f"GPS track for {player}")
        
        # Check shinguard data
        player_dir = f'player_{player_num.replace(".", "_")}'
        if not any(check_file_exists(base_path / f'{player_dir}/{prefix}', True) 
                  for prefix in ['df_trace', 'df_trace_old', 'df_trace_new']):
            warnings.append(f"Shinguard data for {player}")
    
    # Process missing files
    if missing_files:
        error_msg = f"Missing essential files for session {session_id}:\n" + "\n".join(missing_files)
        logging.error(error_msg)
        if strict:
            raise FileNotFoundError(error_msg)
        st.error(error_msg)
        return False
    
    # Process warnings
    if warnings:
        warning_msg = f"Some optional files are missing for session {session_id}:\n" + "\n".join(warnings)
        logging.warning(warning_msg)
        st.warning(warning_msg)
    
    return True


def main():
    try:
        setup_logging()
        available_sessions = get_available_sessions()
        if not available_sessions:
            st.error("No session data found")
            return
            
        selected_session = st.sidebar.selectbox(
            "Select Session",
            available_sessions,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Ensure directories exist
        ensure_session_directories(selected_session)
        
        # Initialize data processor first
        with st.spinner("Initializing application..."):
            data_processor = DataProcessor(selected_session)
            comparison_processor = ComparisonProcessor(data_processor)
            display_manager = DisplayManager(data_processor, comparison_processor)
        
        # Now display session info using the initialized processor
        display_session_info_with_processor(selected_session, data_processor)
        
        # Check files but don't strictly require all to exist
        check_session_files_with_processor(selected_session, data_processor, strict=False)
        
        logging.info(f"Starting analysis for session: {selected_session}")

        st.sidebar.title("Analysis Controls")
        view_type = st.sidebar.radio(
            "Select Analysis Type",
            ["Single Player Analysis", "Multi-Player Comparison", "Hardware Comparison"]  # Added "Hardware Comparison"
        )
        
        # Add visibility controls in sidebar
        st.sidebar.markdown("### Visibility Controls")
        if 'show_gt' not in st.session_state:
            st.session_state.show_gt = True
        if 'show_gpexe' not in st.session_state:
            st.session_state.show_gpexe = True
        if 'show_xseed' not in st.session_state:
            st.session_state.show_xseed = True
            
        st.session_state.show_gt = st.sidebar.checkbox("Show Ground Truth", value=st.session_state.show_gt)
        st.session_state.show_gpexe = st.sidebar.checkbox("Show GPEXE", value=st.session_state.show_gpexe)
        st.session_state.show_xseed = st.sidebar.checkbox("Show Xseed", value=st.session_state.show_xseed)

        if view_type == "Single Player Analysis":
            run_single_player_analysis(data_processor, display_manager, comparison_processor)
        elif view_type == "Multi-Player Comparison":
            run_multi_player_analysis(data_processor, display_manager, comparison_processor)
        elif view_type == "Hardware Comparison":
            # Run our simple hardware comparison
            run_simple_hardware_comparison(data_processor, comparison_processor)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.error("Critical error", exc_info=True)
        st.code(traceback.format_exc())

def run_single_player_analysis(data_processor, display_manager, comparison_processor):
    """Handle single player analysis with hardware type support and consistent sampling."""
    players = data_processor.get_players_list()
    selected_player = st.sidebar.selectbox("Select Player", players)
    
    # Get available hardware types for this player
    available_hw = data_processor.get_available_hardware_types(selected_player)
    if not available_hw:
        st.warning(f"No hardware data found for {selected_player}")
        return
        
    # Add hardware type selector if more than one type is available
    if len(available_hw) > 1:
        hardware_type = st.sidebar.radio(
            "Hardware Type",
            available_hw,
            format_func=lambda x: "New Hardware" if x == "new" else "Old Hardware"
        )
    else:
        hardware_type = available_hw[0]
    
    # Display hardware badge
    display_hardware_badge(hardware_type)
    
    # Get time tolerance from slider
    time_tolerance = st.sidebar.slider(
        "Time Offset (seconds)",
        min_value=-5,
        max_value=5,
        value=0,
        help="Negative values look back in time, positive values look forward",
        key="single_player_time_tolerance"
    )
    
    # Add sampling frequency control
    st.sidebar.subheader("Data Sampling")
    
    # Checkbox to enable/disable resampling
    use_resampling = st.sidebar.checkbox(
        "Enable 10Hz Resampling",
        value=True,
        help="Resample all data to consistent 10Hz frequency"
    )
    
    # If not using resampling, add a warning
    if not use_resampling:
        st.sidebar.warning("Without resampling, accuracy metrics may be affected by differing sampling rates")
    
    # Get protocol info for the selected player and hardware type
    protocol_df = data_processor.get_protocol_info(selected_player, hardware_type)
    
    if protocol_df.empty:
        st.warning(f"No protocol information available for {selected_player} with {hardware_type} hardware")
        return
        
    protocol_id = st.sidebar.selectbox(
        "Select Protocol",
        protocol_df['Protocol ID'].unique(),
        format_func=lambda x: f"Protocol {x}: {protocol_df[protocol_df['Protocol ID'] == x]['Protocol name'].iloc[0]}"
    )

    try:
        # Add debug logging
        logging.info(f"Processing data for {selected_player}, protocol {protocol_id} with {hardware_type} hardware")
        logging.info(f"Resampling enabled: {use_resampling}")
        
        # Get ground truth and GPEXE data
        gt_data, gpexe_data, protocol_info = data_processor.process_player_data(
            selected_player,
            protocol_id,
            time_tolerance,
            hardware_type
        )
        
        # Debug log to see what keys are available in protocol_info
        logging.info(f"Protocol info keys: {list(protocol_info.keys())}")
        
        # Add debug logging
        logging.info(f"Ground truth data points: {len(gt_data)}")
        logging.info(f"GPEXE data points: {len(gpexe_data)}")

        if not gt_data.empty:
            # Handle various possible key names for start and end times
            possible_start_keys = ['Start_Time', 'start_time', 'Start Time', 'StartTime', 'start']
            possible_end_keys = ['End_Time', 'end_time', 'End Time', 'EndTime', 'end']
            
            start_time = None
            for key in possible_start_keys:
                if key in protocol_info:
                    start_time = protocol_info[key]
                    logging.info(f"Found start time using key: {key}")
                    break
            
            end_time = None
            for key in possible_end_keys:
                if key in protocol_info:
                    end_time = protocol_info[key]
                    logging.info(f"Found end time using key: {key}")
                    break
            
            if start_time is None or end_time is None:
                logging.error(f"Could not find start/end time in protocol_info: {list(protocol_info.keys())}")
                st.error("Missing time range information in protocol data")
                return
            
            # Get XSEED data
            xseed_data = data_processor.load_xseed_data(
                selected_player,
                start_time,
                end_time,
                time_tolerance,
                hardware_type
            )
            
            # Log sampling information
            st.subheader("Data Sampling Information")
            sampling_col1, sampling_col2, sampling_col3 = st.columns(3)
            
            with sampling_col1:
                st.write("**Ground Truth**")
                st.write(f"Points: {len(gt_data)}")
                if not gt_data.empty:
                    gt_span = (gt_data['date time'].max() - gt_data['date time'].min()).total_seconds()
                    gt_sampling_rate = len(gt_data) / gt_span if gt_span > 0 else 0
                    st.write(f"Time span: {gt_span:.2f}s")
                    st.write(f"Sampling rate: {gt_sampling_rate:.2f} Hz")
            
            with sampling_col2:
                st.write("**GPEXE Data**") 
                st.write(f"Points: {len(gpexe_data)}")
                if not gpexe_data.empty:
                    gpexe_span = (gpexe_data['date time'].max() - gpexe_data['date time'].min()).total_seconds()
                    gpexe_sampling_rate = len(gpexe_data) / gpexe_span if gpexe_span > 0 else 0
                    st.write(f"Time span: {gpexe_span:.2f}s")
                    st.write(f"Sampling rate: {gpexe_sampling_rate:.2f} Hz")
            
            with sampling_col3:
                st.write("**XSEED Data**")
                st.write(f"Points: {len(xseed_data)}")
                if not xseed_data.empty:
                    xseed_span = (xseed_data['date time'].max() - xseed_data['date time'].min()).total_seconds()
                    xseed_sampling_rate = len(xseed_data) / xseed_span if xseed_span > 0 else 0
                    st.write(f"Time span: {xseed_span:.2f}s")
                    st.write(f"Sampling rate: {xseed_sampling_rate:.2f} Hz")

            tabs = st.tabs(["GPEXE Analysis", "Xseed Analysis", "Combined Analysis"])

            with tabs[0]:
                if not gpexe_data.empty:
                    logging.info("Calculating GPEXE metrics")  # Debug logging
                    metrics = display_manager.calculate_comparison_metrics(gt_data, gpexe_data)
                    display_manager.display_tracking_comparison(
                        ground_truth=gt_data,
                        tracking_data=gpexe_data,
                        metrics=metrics,
                        system_type="GPEXE"
                    )
                else:
                    st.warning("No GPEXE data available")

            with tabs[1]:
                if not xseed_data.empty:
                    metrics = display_manager.calculate_comparison_metrics(gt_data, xseed_data)
                    display_manager.display_tracking_comparison(
                        ground_truth=gt_data,
                        tracking_data=xseed_data,
                        metrics=metrics,
                        system_type=f"Xseed ({hardware_type} hardware)"
                    )
                else:
                    st.warning(f"No Xseed data available for {hardware_type} hardware")

            with tabs[2]:
                if not gpexe_data.empty and not xseed_data.empty:
                    gpexe_metrics = display_manager.calculate_comparison_metrics(gt_data, gpexe_data)
                    xseed_metrics = display_manager.calculate_comparison_metrics(gt_data, xseed_data)
                    display_manager.display_combined_analysis(
                        gt_data,
                        gpexe_data,
                        xseed_data,
                        {'gpexe': gpexe_metrics, 'xseed': xseed_metrics}
                    )
                else:
                    st.warning(f"Both GPEXE and Xseed ({hardware_type} hardware) data required for combined analysis")

    except Exception as e:
        logging.error(f"Error in single player analysis: {str(e)}")
        st.error(f"Error processing data: {str(e)}")
        st.code(traceback.format_exc())

def run_multi_player_analysis(data_processor, display_manager, comparison_processor):
   """Handle multi-player comparison with support for missing data files."""
   protocol_df = data_processor.get_protocol_info('Player 1')
   
   # Check if we have the multi_analysis column (check for column name with and without space)
   multi_analysis_col = None
   for col in protocol_df.columns:
       if col.strip() == 'multi_analysis':
           multi_analysis_col = col
           logging.info(f"Found multi_analysis column: '{col}'")
           break
   
   # Handle case where column isn't found
   if not multi_analysis_col:
       logging.warning("No multi_analysis column found. Available columns: " + 
                     str(protocol_df.columns.tolist()))
       # Default to showing all protocols
       multi_protocols = protocol_df
   else:
       # Debug: Print the unique values in multi_analysis column
       logging.info(f"Debug - multi_analysis unique values: {protocol_df[multi_analysis_col].unique()}")
       
       # Use multi_analysis_col variable instead of direct column name
       try:
           # First try to convert to numeric if it's not already
           protocol_df[multi_analysis_col] = pd.to_numeric(protocol_df[multi_analysis_col], errors='coerce')
           
           # Now filter where multi_analysis is not 0 (treats 1 or any non-zero as True)
           multi_protocols = protocol_df[protocol_df[multi_analysis_col] != 0]
           logging.info(f"Filtered {len(multi_protocols)} multi-player protocols from {len(protocol_df)} total")
       except Exception as e:
           logging.error(f"Error filtering protocols: {str(e)}")
           # Default to all protocols instead of using astype(bool)
           logging.error(f"Will show all protocols")
           multi_protocols = protocol_df
   
   if multi_protocols.empty:
       st.warning("No multi-player protocols available") 
       return

   # Add hardware type selector
   available_hardware = []
   if data_processor.hardware_types['old']:
       available_hardware.append('old')
   if data_processor.hardware_types['new']:
       available_hardware.append('new')
   
   if not available_hardware:
       available_hardware = ['old']  # Default fallback
       
   if len(available_hardware) > 1:
       selected_hardware = st.sidebar.radio(
           "Hardware Type",
           available_hardware,
           format_func=lambda x: "New Hardware" if x == "new" else "Old Hardware"
       )
   else:
       selected_hardware = available_hardware[0]
   
   # Display hardware badge
   display_hardware_badge(selected_hardware)
   
   time_tolerance = st.sidebar.slider(
       "Time Offset (seconds)",
       min_value=-5,
       max_value=5,
       value=0,
       help="Negative values look back in time, positive values look forward",
       key="multi_player_time_tolerance"
   )
   
   protocol_id = st.sidebar.selectbox(
       "Select Protocol",
       multi_protocols['Protocol ID'].unique(),
       format_func=lambda x: f"Protocol {x}: {multi_protocols[multi_protocols['Protocol ID'] == x]['Protocol name'].iloc[0]}"
   )
   
   show_gpexe = st.sidebar.checkbox("Show GPEXE Tracks", value=True)
   show_xseed = st.sidebar.checkbox("Show Xseed Tracks", value=True)
   show_distances = st.sidebar.checkbox("Show Inter-Player Distances", value=True)

   try:
       # Get data for all players
       all_data = {}
       players_with_data = []
       
       # First, collect the available data for each player
       for player in data_processor.get_players_list():
           try:
               gt_data, gpexe_data, protocol_info = data_processor.process_player_data(
                   player, protocol_id, time_tolerance, hardware_type=selected_hardware
               )
               
               # Only include players with valid ground truth data
               if not gt_data.empty:
                   all_data[player] = {
                       'ground_truth': gt_data,
                       'protocol_info': protocol_info
                   }
                   
                   # Add GPEXE data if available
                   if not gpexe_data.empty:
                       all_data[player]['gpexe'] = gpexe_data
                   else:
                       logging.info(f"No GPEXE data available for {player}")
                   
                   # Try to add Xseed data if requested
                   if show_xseed:
                       try:
                           xseed_data = data_processor.load_xseed_data(
                               player,
                               protocol_info['Start_Time'],
                               protocol_info['End_Time'],
                               time_tolerance,
                               hardware_type=selected_hardware
                           )
                           if not xseed_data.empty:
                               all_data[player]['xseed'] = xseed_data
                               logging.info(f"Loaded {len(xseed_data)} Xseed points for {player} with {selected_hardware} hardware")
                           else:
                               logging.info(f"No Xseed data available for {player} with {selected_hardware} hardware")
                       except Exception as xseed_error:
                           logging.warning(f"Error loading Xseed data for {player} with {selected_hardware} hardware: {str(xseed_error)}")
                   
                   # Add player to the list of valid players
                   players_with_data.append(player)
           except Exception as player_error:
               logging.warning(f"Could not process data for {player} with {selected_hardware} hardware: {str(player_error)}")
               continue

       if players_with_data:
           # Create visualization tabs
           tabs = st.tabs(["Combined Tracks", "Distance Analysis"])
           
           with tabs[0]:
               # Create combined plot
               fig = go.Figure()
               colors = {'Player 1': 'blue', 'Player 2': 'green', 'Player 3': 'red'}
               
               # Display hardware type info
               st.markdown(f"**Hardware Type**: {'New Hardware' if selected_hardware == 'new' else 'Old Hardware'}")
               
               # Add ground truth for all players with data
               for player in players_with_data:
                   data = all_data[player]
                   player_color = colors.get(player, 'gray')  # Default to gray if color not defined
                   
                   fig.add_trace(go.Scatter(
                       x=data['ground_truth']['x_real_m'],
                       y=data['ground_truth']['y_real_m'],
                       mode='lines',
                       name=f'{player} Ground Truth',
                       line=dict(color=player_color, width=2, dash='dash')
                   ))
               
               # Plot GPEXE tracks if enabled
               if show_gpexe:
                   for player in players_with_data:
                       data = all_data[player]
                       if 'gpexe' in data and not data['gpexe'].empty:
                           player_color = colors.get(player, 'gray')
                           fig.add_trace(go.Scatter(
                               x=data['gpexe']['x_real_m'],
                               y=data['gpexe']['y_real_m'],
                               mode='markers+lines',
                               name=f'{player} GPEXE',
                               marker=dict(
                                   size=6,
                                   color=data['gpexe'].get('speed', 0),
                                   colorscale='Viridis',
                                   showscale=True,
                                   colorbar=dict(title="Speed (m/s)")
                               ),
                               line=dict(color=player_color, width=1)
                           ))
                       else:
                           st.info(f"No GPEXE data available for {player}")
               
               # Plot Xseed tracks if enabled
               if show_xseed:
                   for player in players_with_data:
                       data = all_data[player]
                       if 'xseed' in data and not data['xseed'].empty:
                           player_color = colors.get(player, 'gray')
                           fig.add_trace(go.Scatter(
                               x=data['xseed']['x_real_m'],
                               y=data['xseed']['y_real_m'],
                               mode='markers+lines',
                               name=f'{player} Xseed ({selected_hardware})',
                               marker=dict(size=6),
                               line=dict(color=player_color, width=1, dash='dot')
                           ))
                       else:
                           st.info(f"No Xseed data available for {player} with {selected_hardware} hardware")

               display_manager.add_pitch_outline(fig)

               fig.update_layout(
                   title=f"Multi-Player Tracking Comparison ({selected_hardware.capitalize()} Hardware)",
                   xaxis_title="X Position (m)",
                   yaxis_title="Y Position (m)",
                   template="plotly_dark",
                   showlegend=True,
                   yaxis=dict(scaleanchor="x", scaleratio=1)
               )

               st.plotly_chart(fig, use_container_width=True)

           with tabs[1]:
               if show_distances and len(players_with_data) >= 2:
                   # Only process distance analysis if we have at least 2 players with data
                   
                   # Calculate distances for each system
                   logging.info("\nProcessing distances for ground truth data:")
                   gt_data_collection = {p: data['ground_truth'] for p, data in all_data.items()}
                   gt_distances = comparison_processor.calculate_inter_player_distances(gt_data_collection)
                   
                   gpexe_distances = {}
                   if show_gpexe:
                       logging.info("\nProcessing distances for GPEXE data:")
                       gpexe_data = {}
                       for p in players_with_data:
                           if 'gpexe' in all_data[p] and not all_data[p]['gpexe'].empty:
                               gpexe_data[p] = all_data[p]['gpexe']
                               logging.info(f"{p}: {len(all_data[p]['gpexe'])} points")
                       
                       if len(gpexe_data) >= 2:  # Need at least 2 players for distances
                           gpexe_distances = comparison_processor.calculate_inter_player_distances(gpexe_data)
                       else:
                           st.warning("GPEXE data available for less than 2 players - can't calculate distances")
                   
                   xseed_distances = {}
                   if show_xseed:
                       logging.info(f"\nProcessing distances for Xseed data with {selected_hardware} hardware:")
                       xseed_data = {}
                       for p in players_with_data:
                           if 'xseed' in all_data[p] and not all_data[p]['xseed'].empty:
                               xseed_data[p] = all_data[p]['xseed']
                               logging.info(f"{p}: {len(all_data[p]['xseed'])} points")
                       
                       if len(xseed_data) >= 2:  # Need at least 2 players for distances
                           xseed_distances = comparison_processor.calculate_inter_player_distances(xseed_data)
                       else:
                           st.warning(f"Xseed data with {selected_hardware} hardware available for less than 2 players - can't calculate distances")
                   
                   # Display distance metrics
                   display_manager.display_distance_metrics_comparison(
                       gt_distances,
                       gpexe_distances if gpexe_distances else None,
                       xseed_distances if xseed_distances else None,
                       hardware_type=selected_hardware
                   )
               elif len(players_with_data) < 2:
                   st.warning("Distance analysis requires at least 2 players with data")
       else:
           st.warning(f"No data available for any player in the selected protocol with {selected_hardware} hardware")

   except Exception as e:
       st.error(f"Error in multi-player analysis: {str(e)}")
       st.code(traceback.format_exc())

def run_system_comparison(data_processor, display_manager, comparison_processor):
    """Handle system comparison analysis."""
    st.subheader("System Comparison Analysis")
    
    # Add time tolerance slider with unique key
    time_tolerance = st.sidebar.slider(
        "Time Offset (seconds)",
        min_value=-5,
        max_value=5,
        value=0,
        help="Negative values look back in time, positive values look forward",
        key="system_comparison_time_tolerance"
    )
    
    # Select protocol and player
    protocol_df = data_processor.get_protocol_info('Player 1')
    protocol_id = st.sidebar.selectbox(
        "Select Protocol",
        protocol_df['Protocol ID'].unique(),
        format_func=lambda x: f"Protocol {x}: {protocol_df[protocol_df['Protocol ID'] == x]['Protocol name'].iloc[0]}"
    )
    
    players = data_processor.get_players_list()
    selected_player = st.sidebar.selectbox("Select Player", players)

    try:
        # Get all tracking data
        gt_data, gpexe_data, protocol_info = data_processor.process_player_data(
            selected_player,
            protocol_id,
            time_tolerance
        )

        xseed_data = data_processor.load_xseed_data(
            selected_player,
            protocol_info['Start_Time'],
            protocol_info['End_Time'],
            time_tolerance
        )

        # Add the debug logging here
        if not gt_data.empty:
            logging.info("\nGround Truth Sample:")
            logging.info(gt_data[['date time', 'x_real_m', 'y_real_m']].head())
            logging.info("\nTracking Data Sample:")
            logging.info(gpexe_data[['date time', 'x_real_m', 'y_real_m']].head())

        if not gt_data.empty:
            comparison_metrics = comparison_processor.analyze_tracking_performance(
                gt_data,
                gpexe_data,
                xseed_data,
                time_tolerance
            )

            display_manager.display_system_comparison(
                gt_data,
                gpexe_data,
                xseed_data,
                comparison_metrics,
                protocol_info,
                selected_player
            )

    except Exception as e:
        st.error(f"Error in system comparison: {str(e)}")
        logging.error(f"System comparison error: {str(e)}", exc_info=True)

def handle_error(error_type: str, error: Exception):
    """Handle different types of errors uniformly."""
    st.error(f"{error_type}: {str(error)}")
    logging.error(f"{error_type}: {str(error)}", exc_info=True)
    
    if st.button("Show Error Details"):
        show_error_details()
    
    if st.button("Restart Application"):
        st.experimental_rerun()

def show_error_details():
    """Display detailed error information and recovery options."""
    st.markdown("""
    ### Error Details
    1. Check if all required files are present in the correct locations
    2. Verify file formats and data integrity
    3. Check the application logs for detailed error messages
    
    ### Recovery Options
    - Ensure all data files use semicolon (;) as separator
    - Verify GPS and shinguard data timestamps are in the correct format
    - Check if stadium properties file contains valid coordinates
    """)
def display_hardware_badge(hardware_type: str):
    """Display a badge indicating hardware type in the UI."""
    if hardware_type == 'new':
        st.markdown('<span style="background-color: #4CAF50; color: white; padding: 3px 8px; '
                   'border-radius: 5px;">NEW HARDWARE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="background-color: #2196F3; color: white; padding: 3px 8px; '
                   'border-radius: 5px;">OLD HARDWARE</span>', unsafe_allow_html=True)
def check_file_exists(file_path, any_extension=True):
    """
    Check if a file exists, optionally trying multiple extensions.
    
    Args:
        file_path: Base file path without extension
        any_extension: Whether to try multiple extensions
        
    Returns:
        True if file exists, False otherwise
    """
    if any_extension:
        # Try multiple extensions
        path = Path(file_path)
        stem = path.parent / path.stem
        
        for ext in ['.csv', '.xlsx', '.xls']:
            if (stem.with_suffix(ext)).exists():
                return True
        return False
    else:
        # Check exact path
        return Path(file_path).exists()

def detect_file_formats(session_path: Path) -> Dict[str, int]:
    """Detect the file formats present in the session directory."""
    formats = {
        'csv': 0,
        'xlsx': 0,
        'xls': 0
    }
    
    # Count files of each format
    for ext in formats.keys():
        formats[ext] = len(list(session_path.glob(f'*.{ext}')))
        
        # Also check subdirectories
        for subdir in session_path.glob('player_*'):
            if subdir.is_dir():
                formats[ext] += len(list(subdir.glob(f'*.{ext}')))
    
    return formats

def display_session_info_with_processor(session_id: str, data_processor: DataProcessor):
    """Display information about the current session using existing DataProcessor."""
    st.sidebar.markdown("### Session Information")
    
    player_list = data_processor.get_players_list()
    player_count = len(player_list)
    player_numbers = [p.split()[-1] for p in player_list]
    
    # Check which types of files exist
    base_path = Path(f'data/{session_id}')
    
    # Detect file formats
    formats = detect_file_formats(base_path)
    
    # Check for Xseed versions - USE DATA_PROCESSOR METHOD
    xseed_old = any(data_processor._check_file_exists(f'player_{p.replace(".", "_")}/df_trace_old') 
                   for p in player_numbers)
    xseed_new = any(data_processor._check_file_exists(f'player_{p.replace(".", "_")}/df_trace_new') 
                   for p in player_numbers)
    xseed_standard = any(data_processor._check_file_exists(f'player_{p.replace(".", "_")}/df_trace') 
                        for p in player_numbers)
    
    # Display info
    st.sidebar.markdown(f"**Session ID:** {session_id}")
    st.sidebar.markdown(f"**Players:** {player_count} ({', '.join(player_list)})")
    st.sidebar.markdown(f"**File Formats:** {formats['csv']} CSV, {formats['xlsx']+formats['xls']} Excel")
    
    gpexe_exists = any(data_processor._check_file_exists(f'gpexe_track_{p}') for p in player_numbers)
    st.sidebar.markdown(f"**GPEXE Data:** {'✅' if gpexe_exists else '❌'}")
    st.sidebar.markdown(f"**Xseed Old HW:** {'✅' if xseed_old else '❌'}")
    st.sidebar.markdown(f"**Xseed New HW:** {'✅' if xseed_new else '❌'}")
    if xseed_standard:
        st.sidebar.markdown(f"**Xseed Standard:** ✅")

def check_session_files_with_processor(session_id: str, data_processor: DataProcessor, strict=False):
    """
    Check if required files exist for the session using existing DataProcessor.
    With strict=False, will only warn about missing files instead of raising errors.
    """
    base_path = Path(f'data/{session_id}')
    player_list = data_processor.get_players_list()
    
    # Check essential files
    missing_files = []
    
    # Stadium properties is essential
    if not any((base_path / f'stadium_properties{ext}').exists() 
              for ext in ['.csv', '.xlsx', '.xls']):
        missing_files.append(f"Stadium properties")
    
    # Check file sets for each detected player
    warnings = []
    for player in player_list:
        player_num = player.split()[-1]
        
        # Check GPS track file
        if not check_file_exists(base_path / f'gpexe_track_{player_num}', True):
            warnings.append(f"GPS track for {player}")
        
        # Check shinguard data
        player_dir = f'player_{player_num.replace(".", "_")}'
        if not any(check_file_exists(base_path / f'{player_dir}/{prefix}', True) 
                  for prefix in ['df_trace', 'df_trace_old', 'df_trace_new']):
            warnings.append(f"Shinguard data for {player}")
    
    # Process missing files
    if missing_files:
        error_msg = f"Missing essential files for session {session_id}:\n" + "\n".join(missing_files)
        logging.error(error_msg)
        if strict:
            raise FileNotFoundError(error_msg)
        st.error(error_msg)
        return False
    
    # Process warnings
    if warnings:
        warning_msg = f"Some optional files are missing for session {session_id}:\n" + "\n".join(warnings)
        logging.warning(warning_msg)
        st.warning(warning_msg)
    
    return True     

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logging.error("Program terminated due to error", exc_info=True)
        st.error("Critical error occurred. Please check the logs.")
        sys.exit(1)
   

