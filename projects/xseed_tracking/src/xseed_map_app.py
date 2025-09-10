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

def display_xseed_map(session_id: str):
    """Main function to display Xseed data on a map for a specific session"""
    st.title(f"Xseed Map Visualization - {session_id}")
    
    # Initialize data processor and display manager using your existing code
    try:
        with st.spinner("Loading session data..."):
            data_processor = DataProcessor(session_id)
            comparison_processor = ComparisonProcessor(data_processor)
            display_manager = DisplayManager(data_processor, comparison_processor)
            
        # Player selection - uses your existing player detection logic
        players = data_processor.get_players_list()
        selected_player = st.sidebar.selectbox("Select Player", players)
        
        # Hardware type selection - uses your existing hardware detection logic
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
        
        # Protocol selection - reuse your existing protocol logic
        protocol_df = data_processor.get_protocol_info(selected_player, hardware_type)
        
        if protocol_df.empty:
            st.warning(f"No protocol information available for {selected_player} with {hardware_type} hardware")
            return
            
        protocol_id = st.sidebar.selectbox(
            "Select Protocol",
            protocol_df['Protocol ID'].unique(),
            format_func=lambda x: f"Protocol {x}: {protocol_df[protocol_df['Protocol ID'] == x]['Protocol name'].iloc[0]}"
        )
        
        # Process player data to get protocol information
        ground_truth, gpexe_data, protocol_info = data_processor.process_player_data(
            selected_player, protocol_id, 0, hardware_type
        )
        
        if not protocol_info:
            st.error(f"No protocol information found for {selected_player} with protocol ID {protocol_id}")
            return
        
        # Extract protocol times
        start_time = protocol_info.get('Start_Time')
        end_time = protocol_info.get('End_Time')
        
        if not start_time or not end_time:
            st.error("Protocol start or end time missing")
            return
        
        # Convert times to datetime for easier manipulation
        base_date = datetime.now().date()
        protocol_start = datetime.combine(base_date, start_time)
        protocol_end = datetime.combine(base_date, end_time)
        protocol_duration_seconds = (protocol_end - protocol_start).total_seconds()
        
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
                selected_start = protocol_start + timedelta(seconds=time_range[0])
                selected_end = protocol_start + timedelta(seconds=time_range[1])
                
                # Display selected time range
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Start: {selected_start.time().strftime('%H:%M:%S')}")
                with col2:
                    st.info(f"End: {selected_end.time().strftime('%H:%M:%S')}")
                
            elif time_selection_method == "Specific Time Range":
                # Allow direct input of start and end times
                col1, col2 = st.columns(2)
                with col1:
                    start_hours = int(start_time.hour)
                    start_minutes = int(start_time.minute)
                    start_seconds = int(start_time.second)
                    
                    input_start_time = st.time_input(
                        "Start time",
                        time(start_hours, start_minutes, start_seconds)
                    )
                    selected_start = datetime.combine(base_date, input_start_time)
                
                with col2:
                    end_hours = int(end_time.hour)
                    end_minutes = int(end_time.minute)
                    end_seconds = int(end_time.second)
                    
                    input_end_time = st.time_input(
                        "End time",
                        time(end_hours, end_minutes, end_seconds)
                    )
                    selected_end = datetime.combine(base_date, input_end_time)
                
                # Validate time range
                if selected_end <= selected_start:
                    st.error("End time must be after start time")
                    return
            
            else:  # Full Protocol
                selected_start = protocol_start
                selected_end = protocol_end
                st.info(f"Full protocol time: {start_time} to {end_time}")
        
        # Time tolerance control
        time_tolerance = st.slider(
            "Time tolerance (seconds)",
            min_value=-5,
            max_value=5,
            value=0,
            help="Adjust time window if needed (negative values look back, positive values look forward)"
        )
        
        # Load Xseed data with the specified time range using your existing data loading function
        with st.spinner("Loading Xseed data..."):
            xseed_data = data_processor.load_xseed_data(
                selected_player,
                selected_start.time(),
                selected_end.time(),
                time_tolerance,
                hardware_type
            )
        
        if xseed_data.empty:
            st.warning(f"No Xseed data found for {selected_player} in the selected time range")
            return
        
        # Display data point count and time range
        st.write(f"Loaded {len(xseed_data)} data points from {selected_start.time()} to {selected_end.time()}")
        
        # Create visualization controls
        st.subheader("Visualization Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_speed = st.checkbox("Show speed data", value=True)
            show_path = st.checkbox("Show continuous path", value=True)
        
        with col2:
            show_animation = st.checkbox("Enable animation", value=True)
            animation_speed = st.slider("Animation speed", min_value=1, max_value=20, value=5)
        
        # Create the map plot
        fig = go.Figure()
        
        # Optionally add continuous path
        if show_path:
            fig.add_trace(go.Scatter(
                x=xseed_data['x_real_m'],
                y=xseed_data['y_real_m'],
                mode='lines',
                name='Path',
                line=dict(
                    color='rgba(0, 150, 150, 0.5)',
                    width=2
                ),
                hoverinfo='none'
            ))
        
        # Add points with color based on speed if available
        if show_speed and 'speed' in xseed_data.columns:
            max_speed = xseed_data['speed'].max()
            fig.add_trace(go.Scatter(
                x=xseed_data['x_real_m'],
                y=xseed_data['y_real_m'],
                mode='markers',
                name='Xseed Data',
                marker=dict(
                    size=8,
                    color=xseed_data['speed'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed (m/s)"),
                    cmin=0,
                    cmax=max_speed
                ),
                text=xseed_data.apply(
                    lambda row: f"Time: {row['date time'].strftime('%H:%M:%S.%f')[:-3]}<br>Speed: {row.get('speed', 0):.2f} m/s",
                    axis=1
                ),
                hoverinfo='text'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=xseed_data['x_real_m'],
                y=xseed_data['y_real_m'],
                mode='markers',
                name='Xseed Data',
                marker=dict(
                    size=8,
                    color='teal' if hardware_type == 'new' else 'green'
                ),
                text=xseed_data['date time'].apply(lambda dt: f"Time: {dt.strftime('%H:%M:%S.%f')[:-3]}"),
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
            title=f"Xseed Tracking Data - {selected_player} ({hardware_type} hardware)",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_dark",
            showlegend=True,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        # Add animation if requested and there are enough data points
        if show_animation and len(xseed_data) > 1:
            try:
                # Prepare frames for animation
                frames = []
                slider_steps = []
                
                # Create a frame for every nth point (based on animation speed)
                step_size = max(1, len(xseed_data) // (100 // animation_speed))
                
                for i in range(0, len(xseed_data), step_size):
                    end_idx = min(i + 1, len(xseed_data))
                    subset = xseed_data.iloc[:end_idx]
                    
                    frame_data = []
                    
                    # Add path up to this point if showing path
                    if show_path:
                        frame_data.append(go.Scatter(
                            x=subset['x_real_m'],
                            y=subset['y_real_m'],
                            mode='lines',
                            name='Path',
                            line=dict(color='rgba(0, 150, 150, 0.5)', width=2),
                            hoverinfo='none'
                        ))
                    
                    # Add points with markers
                    if show_speed and 'speed' in xseed_data.columns:
                        frame_data.append(go.Scatter(
                            x=subset['x_real_m'],
                            y=subset['y_real_m'],
                            mode='markers',
                            name='Xseed Data',
                            marker=dict(
                                size=8,
                                color=subset['speed'] if 'speed' in subset.columns else 'blue',
                                colorscale='Viridis',
                                showscale=True,
                                cmin=0,
                                cmax=max_speed
                            ),
                            hoverinfo='text'
                        ))
                    else:
                        frame_data.append(go.Scatter(
                            x=subset['x_real_m'],
                            y=subset['y_real_m'],
                            mode='markers',
                            name='Xseed Data',
                            marker=dict(
                                size=8,
                                color='teal' if hardware_type == 'new' else 'green'
                            ),
                            hoverinfo='text'
                        ))
                    
                    # Add current position with different color and larger size
                    if not subset.empty:
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
                            name='Current Position',
                            text=f"Time: {current_point['date time'].iloc[0].strftime('%H:%M:%S.%f')[:-3]}",
                            hoverinfo='text'
                        ))
                    
                    # Create frame without trying to add pitch outline
                    frame = go.Frame(
                        data=frame_data,
                        name=f"frame{i}"
                    )
                    frames.append(frame)
                    
                    # Add slider step
                    if not subset.empty:
                        slider_step = {
                            "args": [
                                [f"frame{i}"],
                                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}
                            ],
                            "label": subset.iloc[-1]['date time'].strftime('%H:%M:%S'),
                            "method": "animate"
                        }
                        slider_steps.append(slider_step)
                
                # Add frames to figure
                fig.frames = frames
                
                # Add animation controls if we have slider steps
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
        
        # Display time-specific controls
        st.subheader("Time-Specific Analysis")
        
        # Create a time selector for specific timestamp analysis
        if not xseed_data.empty:
            try:
                # Get timestamp range
                min_time = xseed_data['date time'].min().time()
                max_time = xseed_data['date time'].max().time()
                
                st.write(f"Select a specific timestamp between {min_time} and {max_time}")
                
                # Time selection
                time_options = [dt.strftime('%H:%M:%S.%f')[:-3] for dt in xseed_data['date time']]
                if time_options:
                    selected_timestamp_str = st.select_slider(
                        "Select timestamp",
                        options=time_options,
                        value=time_options[len(time_options)//2]
                    )
                    
                    # Find data for the selected timestamp
                    selected_idx = xseed_data['date time'].apply(
                        lambda dt: dt.strftime('%H:%M:%S.%f')[:-3] == selected_timestamp_str
                    )
                    
                    if any(selected_idx):
                        selected_point = xseed_data[selected_idx].iloc[0]
                        
                        # Display point details
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("X Position", f"{selected_point['x_real_m']:.2f} m")
                            if 'speed' in selected_point:
                                st.metric("Speed", f"{selected_point['speed']:.2f} m/s")
                        
                        with col2:
                            st.metric("Y Position", f"{selected_point['y_real_m']:.2f} m")
                            # Add distance from start if multiple points
                            if len(xseed_data) > 1:
                                start_point = xseed_data.iloc[0]
                                distance_from_start = np.sqrt(
                                    (selected_point['x_real_m'] - start_point['x_real_m'])**2 +
                                    (selected_point['y_real_m'] - start_point['y_real_m'])**2
                                )
                                st.metric("Distance from start", f"{distance_from_start:.2f} m")
                        
                        with col3:
                            # Calculate time from start
                            time_from_start = (selected_point['date time'] - xseed_data['date time'].min()).total_seconds()
                            st.metric("Time from start", f"{time_from_start:.2f} s")
                            
                            # Calculate current percent of total distance
                            if len(xseed_data) > 1:
                                # Find index of selected point
                                idx = xseed_data[selected_idx].index[0]
                                
                                # Calculate total distance
                                distances = np.sqrt(
                                    np.diff(xseed_data['x_real_m'])**2 + 
                                    np.diff(xseed_data['y_real_m'])**2
                                )
                                total_distance = np.sum(distances)
                                
                                # Calculate distance up to this point
                                if idx > 0 and idx - 1 < len(distances):
                                    distance_so_far = np.sum(distances[:idx-1])
                                    if total_distance > 0:
                                        st.metric("Percent of total distance", f"{(distance_so_far/total_distance*100):.1f}%")
            except Exception as time_error:
                st.warning(f"Error in time-specific analysis: {str(time_error)}")
                logging.error(f"Time analysis error: {str(time_error)}", exc_info=True)
        
        # Display statistics for the loaded data
        st.subheader("Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total points", len(xseed_data))
            if 'speed' in xseed_data.columns:
                st.metric("Max speed", f"{xseed_data['speed'].max():.2f} m/s")
        
        with col2:
            # Calculate total distance traveled
            if len(xseed_data) > 1:
                distances = np.sqrt(
                    np.diff(xseed_data['x_real_m'])**2 + 
                    np.diff(xseed_data['y_real_m'])**2
                )
                total_distance = np.sum(distances)
                st.metric("Total distance", f"{total_distance:.2f} m")
            
            if 'speed' in xseed_data.columns:
                st.metric("Average speed", f"{xseed_data['speed'].mean():.2f} m/s")
        
        with col3:
            # Calculate time span
            time_span = (xseed_data['date time'].max() - xseed_data['date time'].min()).total_seconds()
            st.metric("Time span", f"{time_span:.2f} s")
            
            # Calculate sampling rate
            if time_span > 0:
                sampling_rate = len(xseed_data) / time_span
                st.metric("Sampling rate", f"{sampling_rate:.2f} Hz")
        
        # Add data download option
        if not xseed_data.empty and st.checkbox("Show raw data"):
            st.dataframe(xseed_data)
            
            # Add CSV download option
            csv = xseed_data.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"xseed_data_{selected_player}_{protocol_id}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logging.error(f"Error in Xseed map visualization: {str(e)}", exc_info=True)
        st.exception(e)  # Show the full exception for debugging

if __name__ == "__main__":
    # Session selection
    st.sidebar.title("Xseed Map Visualizer")
    available_sessions = get_available_sessions()
    
    if not available_sessions:
        st.error("No session data found")
    else:
        selected_session = st.sidebar.selectbox(
            "Select Session",
            available_sessions,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        display_xseed_map(selected_session)