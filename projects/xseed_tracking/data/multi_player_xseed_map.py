import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import sys
import os
import logging
from pathlib import Path

# Fix the import system by adding src directory to path
# This ensures 'src.' prefix works correctly in imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to path

# Now we can import with src prefix
from src.data_processor import DataProcessor
from src.display_manager import DisplayManager
from src.comparison_processor import ComparisonProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xseed_map.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_available_sessions() -> list:
    """Get list of available sessions from data directory"""
    base_path = Path(parent_dir) / 'data'
    return [d.name for d in base_path.iterdir() 
            if d.is_dir() and d.name != 'output']

def display_multi_player_xseed_map(session_id: str):
    """Main function to display Xseed data on a map for all players simultaneously"""
    st.title(f"Multi-Player Xseed Map Visualization - {session_id}")
    
    # Initialize data processor and display manager
    try:
        with st.spinner("Loading session data..."):
            data_processor = DataProcessor(session_id)
            comparison_processor = ComparisonProcessor(data_processor)
            display_manager = DisplayManager(data_processor, comparison_processor)
            
        # Get all available players
        all_players = data_processor.get_players_list()
        if not all_players:
            st.error("No players found in this session")
            return
            
        # Show available players as a multiselect instead of a single selector
        selected_players = st.sidebar.multiselect(
            "Select Players to Display",
            options=all_players,
            default=all_players[:min(4, len(all_players))]  # Default to first 4 players or less
        )
        
        if not selected_players:
            st.warning("Please select at least one player to display")
            return
            
        # Hardware type selection
        hardware_options = []
        for player in selected_players:
            hw_types = data_processor.get_available_hardware_types(player)
            for hw in hw_types:
                if hw not in hardware_options:
                    hardware_options.append(hw)
        
        if not hardware_options:
            st.warning("No hardware data found for selected players")
            return
            
        # Add hardware type selector
        hardware_type = st.sidebar.radio(
            "Hardware Type",
            options=hardware_options,
            format_func=lambda x: "New Hardware" if x == "new" else "Old Hardware"
        )
        
        # Protocol selection - find protocols available for all selected players
        protocol_options = []
        protocol_info_map = {}
        
        # Get first player's protocols as a starting point
        if selected_players:
            first_player = selected_players[0]
            protocol_df = data_processor.get_protocol_info(first_player, hardware_type)
            
            if not protocol_df.empty:
                protocol_options = protocol_df['Protocol ID'].unique().tolist()
                
                # Create a mapping for protocol IDs to names
                for pid in protocol_options:
                    name_row = protocol_df[protocol_df['Protocol ID'] == pid]
                    if not name_row.empty:
                        protocol_info_map[pid] = name_row['Protocol name'].iloc[0]
            
            # Filter to protocols that are available for all selected players
            for player in selected_players[1:]:
                player_protocol_df = data_processor.get_protocol_info(player, hardware_type)
                if not player_protocol_df.empty:
                    player_protocols = player_protocol_df['Protocol ID'].unique().tolist()
                    # Keep only protocols available for this player too
                    protocol_options = [p for p in protocol_options if p in player_protocols]
        
        if not protocol_options:
            st.warning(f"No common protocols found for the selected players with {hardware_type} hardware")
            return
            
        # Protocol selection dropdown
        protocol_id = st.sidebar.selectbox(
            "Select Protocol",
            options=protocol_options,
            format_func=lambda x: f"Protocol {x}: {protocol_info_map.get(x, 'Unknown')}"
        )
        
        # Create a dictionary to store each player's data
        player_data = {}
        global_start_time = None
        global_end_time = None
        
        # Load data for each selected player
        with st.spinner("Loading player data..."):
            for player in selected_players:
                # Process player data to get protocol information
                ground_truth, gpexe_data, protocol_info = data_processor.process_player_data(
                    player, protocol_id, 0, hardware_type
                )
                
                if not protocol_info:
                    st.warning(f"No protocol information found for {player} with protocol ID {protocol_id}")
                    continue
                
                # Extract protocol times
                start_time = protocol_info.get('Start_Time')
                end_time = protocol_info.get('End_Time')
                
                if not start_time or not end_time:
                    st.warning(f"Protocol start or end time missing for {player}")
                    continue
                
                # Convert times to datetime for easier manipulation
                base_date = datetime.now().date()
                protocol_start = datetime.combine(base_date, start_time)
                protocol_end = datetime.combine(base_date, end_time)
                
                # Track global start and end times for all players
                if global_start_time is None or protocol_start < global_start_time:
                    global_start_time = protocol_start
                
                if global_end_time is None or protocol_end > global_end_time:
                    global_end_time = protocol_end
                
                # Store player data
                player_data[player] = {
                    'protocol_info': protocol_info,
                    'start_time': protocol_start,
                    'end_time': protocol_end,
                    'ground_truth': ground_truth
                }
        
        if not player_data:
            st.error("No valid player data found for the selected protocol")
            return
            
        if global_start_time is None or global_end_time is None:
            st.error("Could not determine global time range")
            return
            
        # Calculate global protocol duration
        protocol_duration_seconds = (global_end_time - global_start_time).total_seconds()
        
        st.subheader("Time Range Selection")
        
        # Create expandable section for advanced time controls
        with st.expander("Time Range Controls", expanded=True):
            # Method selection for time range
            time_selection_method = st.radio(
                "Time Selection Method",
                ["Slider", "Specific Time Range", "Full Protocol"],
                horizontal=True
            )
            
            if time_selection_method == "Slider":
                # Time range slider that shows actual timestamps
                time_range = st.slider(
                    "Select time range",
                    min_value=0.0,
                    max_value=protocol_duration_seconds,
                    value=(0.0, protocol_duration_seconds),
                    step=1.0,
                    format="%.1f s"
                )
                
                # Convert slider values (seconds from start) to actual timestamps
                selected_start = global_start_time + timedelta(seconds=time_range[0])
                selected_end = global_start_time + timedelta(seconds=time_range[1])
                
                # Display selected time range
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Start: {selected_start.time().strftime('%H:%M:%S')}")
                with col2:
                    st.info(f"End: {selected_end.time().strftime('%H:%M:%S')}")
                
            elif time_selection_method == "Specific Time Range":
                # Allow direct input of start and end times with text inputs for more precision
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Start Time")
                    start_time_input = st.text_input(
                        "Enter start time (HH:MM:SS.ms)",
                        value=global_start_time.strftime('%H:%M:%S.000')
                    )
                    
                    try:
                        # Parse the start time input
                        start_parts = start_time_input.split(':')
                        if len(start_parts) == 3:
                            # Check if there are milliseconds
                            second_parts = start_parts[2].split('.')
                            start_hour = int(start_parts[0])
                            start_minute = int(start_parts[1])
                            
                            if len(second_parts) == 2:
                                start_second = int(second_parts[0])
                                start_millisecond = int(second_parts[1].ljust(3, '0')[:3])
                            else:
                                start_second = int(start_parts[2])
                                start_millisecond = 0
                                
                            input_start_time = time(start_hour, start_minute, start_second, start_millisecond * 1000)
                            selected_start = datetime.combine(base_date, input_start_time)
                        else:
                            st.error("Invalid start time format")
                            return
                    except ValueError as e:
                        st.error(f"Invalid start time: {str(e)}")
                        return
                
                with col2:
                    st.subheader("End Time")
                    end_time_input = st.text_input(
                        "Enter end time (HH:MM:SS.ms)",
                        value=global_end_time.strftime('%H:%M:%S.000')
                    )
                    
                    try:
                        # Parse the end time input
                        end_parts = end_time_input.split(':')
                        if len(end_parts) == 3:
                            # Check if there are milliseconds
                            second_parts = end_parts[2].split('.')
                            end_hour = int(end_parts[0])
                            end_minute = int(end_parts[1])
                            
                            if len(second_parts) == 2:
                                end_second = int(second_parts[0])
                                end_millisecond = int(second_parts[1].ljust(3, '0')[:3])
                            else:
                                end_second = int(end_parts[2])
                                end_millisecond = 0
                                
                            input_end_time = time(end_hour, end_minute, end_second, end_millisecond * 1000)
                            selected_end = datetime.combine(base_date, input_end_time)
                        else:
                            st.error("Invalid end time format")
                            return
                    except ValueError as e:
                        st.error(f"Invalid end time: {str(e)}")
                        return
                
                # Validate time range
                if selected_end <= selected_start:
                    st.error("End time must be after start time")
                    return
                
                # Display the selected time range with millisecond precision
                st.success(f"Selected time range: {selected_start.strftime('%H:%M:%S.%f')[:-3]} to {selected_end.strftime('%H:%M:%S.%f')[:-3]}")
        
        # Time tolerance control
        time_tolerance = st.slider(
            "Time tolerance (seconds)",
            min_value=-5,
            max_value=5,
            value=0,
            help="Adjust time window if needed (negative values look back, positive values look forward)"
        )
        
        # Load Xseed data for all selected players with the specified time range
        with st.spinner("Loading Xseed data for all players..."):
            for player in selected_players:
                # Load Xseed data using your existing data loading function
                xseed_data = data_processor.load_xseed_data(
                    player,
                    selected_start.time(),
                    selected_end.time(),
                    time_tolerance,
                    hardware_type
                )
                
                if not xseed_data.empty:
                    player_data[player]['xseed_data'] = xseed_data
                    st.write(f"Loaded {len(xseed_data)} data points for {player}")
                else:
                    st.warning(f"No Xseed data found for {player} in the selected time range")
        
        # Check if we have any Xseed data to display
        players_with_data = [p for p in player_data if 'xseed_data' in player_data[p] and not player_data[p]['xseed_data'].empty]
        
        if not players_with_data:
            st.error("No Xseed data found for any selected player in the specified time range")
            return
            
        # Create visualization controls
        st.subheader("Visualization Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_speed = st.checkbox("Show speed data", value=True)
            show_path = st.checkbox("Show continuous path", value=True)
        
        with col2:
            show_ground_truth = st.checkbox("Show ground truth", value=False)
            show_animation = st.checkbox("Enable animation", value=True)
            if show_animation:
                animation_speed = st.slider("Animation speed", min_value=1, max_value=20, value=5)
        
        # Define player colors (ensure we have distinct colors for each player)
        player_colors = {
            'Player 1': {'path': 'rgba(0, 150, 150, 0.5)', 'marker': 'teal' if hardware_type == 'new' else 'green'},
            'Player 2': {'path': 'rgba(150, 0, 150, 0.5)', 'marker': 'purple'},
            'Player 3': {'path': 'rgba(150, 150, 0, 0.5)', 'marker': 'gold'},
            'Player 4': {'path': 'rgba(150, 0, 0, 0.5)', 'marker': 'red'},
            'Player 5': {'path': 'rgba(0, 0, 150, 0.5)', 'marker': 'blue'},
            'Player 6': {'path': 'rgba(0, 150, 0, 0.5)', 'marker': 'lime'},
        }
        
        # Create the map plot
        fig = go.Figure()
        
        # Add data for each player
        for player in players_with_data:
            xseed_data = player_data[player]['xseed_data']
            
            # Get player color (default to a standard color if not in the map)
            player_color = player_colors.get(player, {'path': 'rgba(100, 100, 100, 0.5)', 'marker': 'gray'})
            
            # Optionally add continuous path
            if show_path:
                fig.add_trace(go.Scatter(
                    x=xseed_data['x_real_m'],
                    y=xseed_data['y_real_m'],
                    mode='lines',
                    name=f'{player} Path',
                    line=dict(
                        color=player_color['path'],
                        width=2
                    ),
                    hoverinfo='none'
                ))
            
            # Add ground truth if available and requested
            if show_ground_truth and 'ground_truth' in player_data[player] and not player_data[player]['ground_truth'].empty:
                gt_data = player_data[player]['ground_truth']
                fig.add_trace(go.Scatter(
                    x=gt_data['x_real_m'],
                    y=gt_data['y_real_m'],
                    mode='lines',
                    name=f'{player} Ground Truth',
                    line=dict(
                        color='white',
                        width=1,
                        dash='dot'
                    ),
                    hoverinfo='none'
                ))
            
            # Add points with color based on speed if available
            if show_speed and 'speed' in xseed_data.columns:
                fig.add_trace(go.Scatter(
                    x=xseed_data['x_real_m'],
                    y=xseed_data['y_real_m'],
                    mode='markers',
                    name=f'{player} Data',
                    marker=dict(
                        size=8,
                        color=xseed_data['speed'],
                        colorscale='Viridis',
                        showscale=True if player == players_with_data[0] else False,  # Only show colorbar for first player
                        colorbar=dict(title="Speed (m/s)") if player == players_with_data[0] else None,
                        cmin=0,
                        cmax=max([player_data[p]['xseed_data']['speed'].max() 
                                for p in players_with_data if 'speed' in player_data[p]['xseed_data'].columns], default=5)
                    ),
                    text=xseed_data.apply(
                        lambda row: f"{player}<br>Time: {row['date time'].strftime('%H:%M:%S.%f')[:-3]}<br>Speed: {row.get('speed', 0):.2f} m/s",
                        axis=1
                    ),
                    hoverinfo='text'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=xseed_data['x_real_m'],
                    y=xseed_data['y_real_m'],
                    mode='markers',
                    name=f'{player} Data',
                    marker=dict(
                        size=8,
                        color=player_color['marker']
                    ),
                    text=xseed_data.apply(
                        lambda row: f"{player}<br>Time: {row['date time'].strftime('%H:%M:%S.%f')[:-3]}",
                        axis=1
                    ),
                    hoverinfo='text'
                ))
        
        # Add pitch outline using your existing function - WITH ERROR HANDLING
        try:
            display_manager.add_pitch_outline(fig)
        except Exception as pitch_error:
            st.warning(f"Could not add pitch outline: {str(pitch_error)}")
            logging.error(f"Pitch outline error: {str(pitch_error)}", exc_info=True)
        
        # Update layout
        fig.update_layout(
            title=f"Multi-Player Xseed Tracking - {hardware_type.capitalize()} Hardware",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_dark",
            showlegend=True,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        # Add animation if requested and there are enough data points
        if show_animation:
            try:
                # Prepare frames for animation
                frames = []
                slider_steps = []
                
                # Find common time points across all players
                all_timestamps = []
                for player in players_with_data:
                    player_times = player_data[player]['xseed_data']['date time'].tolist()
                    all_timestamps.extend(player_times)
                
                # Sort all timestamps and remove duplicates
                unique_timestamps = sorted(list(set(all_timestamps)))
                
                # Create a frame for every nth timestamp (based on animation speed)
                step_size = max(1, len(unique_timestamps) // (100 // animation_speed))
                
                for i in range(0, len(unique_timestamps), step_size):
                    current_time = unique_timestamps[i]
                    frame_data = []
                    
                    # For each player, add data up to this timestamp
                    for player in players_with_data:
                        xseed_data = player_data[player]['xseed_data']
                        player_color = player_colors.get(player, {'path': 'rgba(100, 100, 100, 0.5)', 'marker': 'gray'})
                        
                        # Filter data up to current timestamp
                        mask = xseed_data['date time'] <= current_time
                        subset = xseed_data[mask]
                        
                        if not subset.empty:
                            # Add path up to this point if showing path
                            if show_path:
                                frame_data.append(go.Scatter(
                                    x=subset['x_real_m'],
                                    y=subset['y_real_m'],
                                    mode='lines',
                                    name=f'{player} Path',
                                    line=dict(
                                        color=player_color['path'],
                                        width=2
                                    ),
                                    hoverinfo='none'
                                ))
                            
                            # Add points with markers
                            if show_speed and 'speed' in subset.columns:
                                frame_data.append(go.Scatter(
                                    x=subset['x_real_m'],
                                    y=subset['y_real_m'],
                                    mode='markers',
                                    name=f'{player} Data',
                                    marker=dict(
                                        size=8,
                                        color=subset['speed'],
                                        colorscale='Viridis',
                                        showscale=True if player == players_with_data[0] else False,
                                        cmin=0,
                                        cmax=max([player_data[p]['xseed_data']['speed'].max() 
                                                for p in players_with_data if 'speed' in player_data[p]['xseed_data'].columns], default=5)
                                    ),
                                    hoverinfo='text'
                                ))
                            else:
                                frame_data.append(go.Scatter(
                                    x=subset['x_real_m'],
                                    y=subset['y_real_m'],
                                    mode='markers',
                                    name=f'{player} Data',
                                    marker=dict(
                                        size=8,
                                        color=player_color['marker']
                                    ),
                                    hoverinfo='text'
                                ))
                            
                            # Add current position with different color and larger size
                            current_point = subset.iloc[-1:]
                            frame_data.append(go.Scatter(
                                x=[current_point['x_real_m'].iloc[0]],
                                y=[current_point['y_real_m'].iloc[0]],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color='yellow',
                                    line=dict(width=2, color='black')
                                ),
                                name=f'{player} Current',
                                text=f"{player}<br>Time: {current_point['date time'].iloc[0].strftime('%H:%M:%S.%f')[:-3]}",
                                hoverinfo='text'
                            ))
                    
                    # Create frame
                    frame = go.Frame(
                        data=frame_data,
                        name=f"frame{i}"
                    )
                    frames.append(frame)
                    
                    # Add slider step with timestamp
                    slider_step = {
                        "args": [
                            [f"frame{i}"],
                            {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}
                        ],
                        "label": current_time.strftime('%H:%M:%S'),
                        "method": "animate"
                    }
                    slider_steps.append(slider_step)
                
                # Add frames to figure
                fig.frames = frames
                
                # Add animation controls
                if slider_steps:
                    fig.update_layout(
                        updatemenus=[
                            {
                                "buttons": [
                                    {
                                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                                        "label": "Play",
                                        "method": "animate"
                                    },
                                    {
                                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                                        "label": "Pause",
                                        "method": "animate"
                                    }
                                ],
                                "direction": "left",
                                "pad": {"r": 10, "t": 10},
                                "type": "buttons",
                                "x": 0.1,
                                "y": 0
                            }
                        ],
                        sliders=[
                            {
                                "active": 0,
                                "yanchor": "top",
                                "xanchor": "left",
                                "currentvalue": {
                                    "font": {"size": 16},
                                    "prefix": "Time: ",
                                    "visible": True,
                                    "xanchor": "right"
                                },
                                "transition": {"duration": 50, "easing": "cubic-in-out"},
                                "pad": {"b": 10, "t": 50},
                                "len": 0.9,
                                "x": 0.1,
                                "y": 0,
                                "steps": slider_steps
                            }
                        ]
                    )
            except Exception as anim_error:
                st.warning(f"Could not create animation: {str(anim_error)}")
                logging.error(f"Animation error: {str(anim_error)}", exc_info=True)
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics for each player
        st.subheader("Player Statistics")
        
        # Create statistics table for all players
        stat_cols = st.columns(len(players_with_data))
        
        for i, player in enumerate(players_with_data):
            with stat_cols[i]:
                st.subheader(player)
                xseed_data = player_data[player]['xseed_data']
                
                # Calculate statistics
                st.metric("Total points", len(xseed_data))
                
                if 'speed' in xseed_data.columns:
                    st.metric("Max speed", f"{xseed_data['speed'].max():.2f} m/s")
                    st.metric("Average speed", f"{xseed_data['speed'].mean():.2f} m/s")
                
                # Calculate total distance traveled
                if len(xseed_data) > 1:
                    distances = np.sqrt(
                        np.diff(xseed_data['x_real_m'])**2 + 
                        np.diff(xseed_data['y_real_m'])**2
                    )
                    total_distance = np.sum(distances)
                    st.metric("Total distance", f"{total_distance:.2f} m")
                
                # Calculate time span
                time_span = (xseed_data['date time'].max() - xseed_data['date time'].min()).total_seconds()
                st.metric("Time span", f"{time_span:.2f} s")
                
                # Calculate sampling rate
                if time_span > 0:
                    sampling_rate = len(xseed_data) / time_span
                    st.metric("Sampling rate", f"{sampling_rate:.2f} Hz")
        
        # Add data download options for each player
        with st.expander("Download Player Data"):
            for player in players_with_data:
                xseed_data = player_data[player]['xseed_data']
                
                if not xseed_data.empty:
                    # Add CSV download option
                    csv = xseed_data.to_csv(index=False)
                    st.download_button(
                        label=f"Download {player} data as CSV",
                        data=csv,
                        file_name=f"xseed_data_{player}_{protocol_id}.csv",
                        mime="text/csv",
                        key=f"download_{player}"  # Unique key for each button
                    )
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logging.error(f"Error in multi-player Xseed map visualization: {str(e)}", exc_info=True)
        st.exception(e)  # Show the full exception for debugging

if __name__ == "__main__":
    # Session selection
    st.sidebar.title("Multi-Player Xseed Map")
    available_sessions = get_available_sessions()
    
    if not available_sessions:
        st.error("No session data found")
    else:
        selected_session = st.sidebar.selectbox(
            "Select Session",
            available_sessions,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        display_multi_player_xseed_map(selected_session)