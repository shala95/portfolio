import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import re
import os
import glob
from datetime import datetime, timedelta
from player_data_analysis import PlayerDataAnalyzer
import time
import tempfile


def transform_from_xseed_to_opta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform coordinates from XSeed format to Opta format.
    Inverts both X and Y coordinates (100 - coordinate).
    
    Args:
        df: DataFrame containing coordinate columns
    
    Returns:
        DataFrame with transformed coordinates
    """
    # List of coordinate columns to transform
    coordinate_cols = [
        ('x_ma_3', 'y_ma_3')
    ]
    
    # Transform each pair of coordinates if they exist
    for x_col, y_col in coordinate_cols:
        if x_col in df.columns and y_col in df.columns:
            # Store original values
            x_orig = df[x_col].copy()
            y_orig = df[y_col].copy()
            
            # Swap and invert
            df[x_col] =  y_orig
            df[y_col] =  x_orig
    
    return df



# Define the SoccermentPalette class for colors
class SoccermentPalette:
    def __init__(self):
        self.deep_navy = '#1B2C3B'    # color1
        self.dark_blue = '#1B567B'    # color2
        self.bright_blue = '#1C82BE'  # color3
        self.sky_blue = '#2CB7D9'     # color4
        self.light_sky_blue = '#43D9FF' # color5
        self.pale_blue = '#99E3F5'    # color6

        self.medium_blue = '#1BB3D7'  # mblue
        self.steel_blue = '#1D83BF'   # sblue
        self.medium_green = '#2DE598' # mgreen
        self.sea_green = '#1DBF7B'    # sgreen
        self.off_white = '#EDEDED'    # mwhite
        self.light_grey = '#BEBEBE'   # mgrey
        self.charcoal = '#181F2A'     # backg

        # Additional colors for impact categories
        self.pass_color = self.dark_blue
        self.long_ball_color = self.sky_blue
        self.shot_color = self.medium_green
        self.cross_color = self.steel_blue
        self.other_color = self.light_grey
        self.jump_color = self.pale_blue

# Function to create a soccer pitch layout with Plotly
def create_pitch_layout():
    palette = SoccermentPalette()
    
    fig = go.Figure()
    
    # Add the pitch rectangle (main field)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=100, y1=100,
        line=dict(color=palette.off_white),
        fillcolor=palette.deep_navy,
        layer="below"
    )
    
    # Add center line
    fig.add_shape(
        type="line",
        x0=50, y0=0, x1=50, y1=100,
        line=dict(color=palette.off_white)
    )
    
    # Add center circle
    fig.add_shape(
        type="circle",
        x0=40, y0=40, x1=60, y1=60,
        line=dict(color=palette.off_white)
    )
    
    # Add penalty areas
    fig.add_shape(
        type="rect",
        x0=0, y0=21.1, x1=16.5, y1=78.9,
        line=dict(color=palette.off_white)
    )
    fig.add_shape(
        type="rect",
        x0=83.5, y0=21.1, x1=100, y1=78.9,
        line=dict(color=palette.off_white)
    )
    
    # Set layout properties
    fig.update_layout(
        xaxis=dict(
            range=[-5, 105],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-5, 105],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1  # Keep the field proportional
        ),
        width=800,
        height=600,
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(
            font=dict(color=palette.off_white),
            bgcolor=palette.charcoal,
            bordercolor=palette.off_white
        )
    )
    
    return fig
def preprocess_player_data(df, max_interpolation_gap='10s'):
    """
    Preprocess player tracking data to handle missing values
    
    Args:
        df (DataFrame): Player tracking dataframe
        max_interpolation_gap (str): Maximum time gap to interpolate over
        
    Returns:
        DataFrame: Preprocessed dataframe with filled/interpolated values
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Sort by timestamp
    processed_df = processed_df.sort_values('timestamp')
    
    # Forward fill missing x,y values for small gaps
    processed_df['x'] = processed_df['x'].fillna(method='ffill')
    processed_df['y'] = processed_df['y'].fillna(method='ffill')
    
    # Create a regular time series with 100ms frequency
    if not processed_df.empty:
        start_time = processed_df['timestamp'].min()
        end_time = processed_df['timestamp'].max()
        
        # Create regular timestamps
        regular_times = pd.date_range(start=start_time, end=end_time, freq='100ms')
        
        # Create a dataframe with regular timestamps
        regular_df = pd.DataFrame({'timestamp': regular_times})
        
        # Merge with original data
        merged_df = pd.merge_asof(
            regular_df.sort_values('timestamp'),
            processed_df.sort_values('timestamp'),
            on='timestamp', 
            direction='nearest',
            tolerance=pd.Timedelta(max_interpolation_gap)
        )
        
        # Interpolate missing values
        for col in ['x', 'y', 'gps_speed_mean']:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].interpolate(method='linear', limit=100)
        
        return merged_df
    
    return processed_df
    
# Class for managing player tracking data 
class PlayerDataManager:
    """Class for managing player tracking data"""
    def __init__(self, data_folder="./outputs/fetched_data"):
        self.data_folder = data_folder
        self.players_data = {}
        self.global_min_time = None
        self.global_max_time = None
        self.loaded = False
        self.unavailable_players = set()  # Track unavailable players
        self.missing_players_info = {}  # Track players missing from visualization
        
    @st.cache_data
    
    def analyze_multi_session_players(self):
        """Analyze players who appear in multiple sessions
        
        Returns:
            dict: Information about multi-session players
        """
        multi_session_players = {}
        
        for player, player_info in self.players_data.items():
            if player_info.get('multi_session', False):
                df = player_info['data']
                
                # Group data by session
                session_data = {}
                for session_id, session_df in df.groupby('session_id'):
                    session_data[session_id] = {
                        'start_time': session_df['timestamp'].min(),
                        'end_time': session_df['timestamp'].max(),
                        'duration_minutes': (session_df['timestamp'].max() - session_df['timestamp'].min()).total_seconds() / 60,
                        'data_points': len(session_df)
                    }
                
                # Check for time gap between sessions
                session_ids = sorted(session_data.keys())
                session_gaps = []
                
                for i in range(1, len(session_ids)):
                    prev_session = session_data[session_ids[i-1]]
                    curr_session = session_data[session_ids[i]]
                    
                    gap_seconds = (curr_session['start_time'] - prev_session['end_time']).total_seconds()
                    
                    session_gaps.append({
                        'from_session': session_ids[i-1],
                        'to_session': session_ids[i],
                        'gap_minutes': gap_seconds / 60,
                        'from_end': prev_session['end_time'],
                        'to_start': curr_session['start_time']
                    })
                
                multi_session_players[player] = {
                    'id': player_info['id'],
                    'num_sessions': len(session_data),
                    'sessions': session_data,
                    'session_gaps': session_gaps,
                    'total_data_points': len(df)
                }
        
        return multi_session_players

    def load_all_player_data(_self):
        """Load all player CSV files from the data folder with improved handling of multiple sessions
        
        Returns:
            tuple: (player data dict, global min time, global max time)
        """
        all_files = glob.glob(os.path.join(_self.data_folder, "player_*.csv"))
        st.info(f"Found {len(all_files)} player tracking files")
        
        # Group files by player ID first
        player_file_groups = {}
        for file in all_files:
            filename = os.path.basename(file)
            match = re.search(r'player_(\d+)_session_(\d+)_segment_(\d+)', filename)
            if match:
                player_id = match.group(1)
                if player_id not in player_file_groups:
                    player_file_groups[player_id] = []
                player_file_groups[player_id].append(file)
        
        players_data = {}
        global_min_time = None
        global_max_time = None
        unavailable_players = set()
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Available colors for players
        palette = SoccermentPalette()
        colors = [
            palette.bright_blue,
            palette.sky_blue,
            palette.medium_green,
            palette.sea_green,
            palette.steel_blue,
            palette.light_sky_blue,
            palette.medium_blue,
            palette.pale_blue
        ]
        
        # Process each player's files
        for i, (player_id, files) in enumerate(player_file_groups.items()):
            # Update progress
            progress_bar.progress((i + 1) / len(player_file_groups))
            
            player_key = f"Player {player_id}"
            combined_df = pd.DataFrame()
            session_segments = []
            
            for file in files:
                filename = os.path.basename(file)
                try:
                    # Extract session and segment IDs
                    match = re.search(r'player_(\d+)_session_(\d+)_segment_(\d+)', filename)
                    if not match:
                        st.warning(f"Couldn't extract IDs from {filename}")
                        continue
                        
                    session_id = match.group(2)
                    segment_id = match.group(3)
                    session_segments.append((session_id, segment_id))
                    
                    # Read the CSV file
                    df = pd.read_csv(file, low_memory=False)
                    
                    # Convert gps_time to datetime
                    if 'gps_time' in df.columns:
                        try:
                            df['timestamp'] = pd.to_datetime(df['gps_time'], format='mixed', errors='coerce')
                            df = df.dropna(subset=['timestamp'])
                            
                            if df.empty:
                                st.error(f"All timestamps in {filename} were invalid after parsing")
                                continue
                            
                            # Add session and segment info to identify the source
                            df['session_id'] = session_id
                            df['segment_id'] = segment_id
                            
                            # Transform coordinates using the new function
                            df = transform_from_xseed_to_opta(df)
                            
                            # Keep only necessary columns
                            columns_to_keep = ['timestamp', 'x_ma_3', 'y_ma_3', 'gps_speed_mean', 'period', 'session_id', 'segment_id']
                            for col in columns_to_keep:
                                if col not in df.columns and col not in ['timestamp', 'session_id', 'segment_id']:
                                    df[col] = None
                            
                            df = df[columns_to_keep].copy()
                            
                            # Rename coordinate columns to match existing code
                            df = df.rename(columns={'x_ma_3': 'x', 'y_ma_3': 'y'})
                            
                            # Update global time range
                            min_time = df['timestamp'].min()
                            max_time = df['timestamp'].max()
                            
                            if global_min_time is None or min_time < global_min_time:
                                global_min_time = min_time
                                
                            if global_max_time is None or max_time > global_max_time:
                                global_max_time = max_time
                            
                            # Append to combined data
                            combined_df = pd.concat([combined_df, df], ignore_index=True)
                            
                        except Exception as date_error:
                            st.error(f"Error parsing dates in {filename}: {date_error}")
                    
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
            
            # If we have data for this player
            if not combined_df.empty:
                # Sort by timestamp
                combined_df = combined_df.sort_values('timestamp')
                
                # Apply preprocessing to each section individually to avoid incorrect interpolation
                processed_sections = []
                for (session_id, segment_id), section_df in combined_df.groupby(['session_id', 'segment_id']):
                    processed_df = preprocess_player_data(section_df, max_interpolation_gap='10s')
                    processed_sections.append(processed_df)
                
                # Recombine processed sections
                final_df = pd.concat(processed_sections, ignore_index=True).sort_values('timestamp')
                
                # Assign a color to this player
                color = colors[i % len(colors)]
                
                # Store in our dictionary with additional metadata
                players_data[player_key] = {
                    'data': final_df,
                    'id': player_id,
                    'sessions': session_segments,
                    'color': color,
                    'first_appearance': final_df['timestamp'].min(),
                    'last_appearance': final_df['timestamp'].max(),
                    'multi_session': len(session_segments) > 1
                }
            else:
                unavailable_players.add(player_key)
        
        _self.players_data = players_data
        _self.global_min_time = global_min_time
        _self.global_max_time = global_max_time
        _self.unavailable_players = unavailable_players
        _self.loaded = True
        
        return players_data, global_min_time, global_max_time
    
    def get_current_positions(self, current_time, time_threshold=5, force_display_all=True):
        """Get player positions with improved multi-session handling
        
        Args:
            current_time (datetime): Current time
            time_threshold (int): Maximum time difference in seconds to consider a player visible
            force_display_all (bool): Force display of all players regardless of time
            
        Returns:
            dict: Dictionary of player positions and metadata
        """
        positions = {}
        missing_players = {}
        player_session_info = {}
        
        # Convert reference time to pandas timestamp
        ref_ts = pd.Timestamp(current_time)
        
        # Track active players
        active_players = []
        
        for player, player_info in self.players_data.items():
            df = player_info['data']
            is_multi_session = player_info.get('multi_session', False)
            
            # Skip if dataframe is empty
            if df.empty:
                missing_players[player] = {'reason': 'No data available', 'time_diff': float('inf')}
                continue
            
            # First, try to find a point close to the requested time
            try:
                # Calculate time differences for all points
                df['time_diff'] = abs(df['timestamp'] - ref_ts)
                
                # Find the minimum time difference
                min_diff_idx = df['time_diff'].idxmin()
                min_diff_row = df.loc[min_diff_idx]
                min_diff_seconds = min_diff_row['time_diff'].total_seconds()
                
                # Determine if this point is considered active
                is_active = min_diff_seconds <= time_threshold
                
                # Get current session info if available
                current_session = None
                if 'session_id' in min_diff_row:
                    current_session = min_diff_row['session_id']
                    current_segment = min_diff_row['segment_id'] if 'segment_id' in min_diff_row else None
                
                # If we have a valid position and either it's active or we're forcing display
                if pd.notna(min_diff_row['x']) and pd.notna(min_diff_row['y']) and (is_active or force_display_all):
                    positions[player] = {
                        'x': min_diff_row['x'],
                        'y': min_diff_row['y'],
                        'timestamp': min_diff_row['timestamp'],
                        'speed': min_diff_row['gps_speed_mean'] if pd.notna(min_diff_row['gps_speed_mean']) else 0,
                        'color': player_info['color'],
                        'id': player_info['id'],
                        'time_diff': min_diff_seconds,
                        'active': is_active,
                        'session_id': current_session,
                        'segment_id': current_segment,
                        'multi_session_player': is_multi_session
                    }
                    
                    # Track active players
                    if is_active:
                        active_players.append(player)
                    
                    # Store session info for this player
                    if current_session:
                        player_session_info[player] = {
                            'current_session': current_session,
                            'current_segment': current_segment,
                            'all_sessions': player_info.get('sessions', [])
                        }
                else:
                    # If position is invalid or player isn't active and we're not forcing display
                    missing_players[player] = {
                        'reason': 'No valid position data within threshold' if is_active else f'Time difference too large: {min_diff_seconds:.1f}s',
                        'time_diff': min_diff_seconds
                    }
            except Exception as e:
                missing_players[player] = {'reason': f'Error finding position: {str(e)}', 'time_diff': float('inf')}
        
        # Store active players and other info
        self.active_players = active_players
        self.missing_players_info = missing_players
        self.player_session_info = player_session_info
        
        return positions
    
    def get_trail_positions(self, current_time, trail_duration=5, time_threshold=None, specific_players=None):
        """Get player trail positions for a time window
        
        Args:
            current_time (datetime): Current time 
            trail_duration (int): Duration of trail in seconds
            time_threshold (int, optional): Maximum time difference to consider when determining visible players
            specific_players (list, optional): List of player keys to process
            
        Returns:
            dict: Dictionary mapping player keys to their trail data
        """
        trail_data = {}
        trail_start_time = current_time - timedelta(seconds=trail_duration)
        
        # Get players to process
        if specific_players is not None:
            players_to_process = specific_players
        elif time_threshold is not None:
            # Get visible players based on current time threshold if provided
            visible_players = self.get_current_positions(current_time, time_threshold).keys()
            players_to_process = visible_players
        else:
            players_to_process = self.players_data.keys()
        
        # Only process trails for requested players
        for player in players_to_process:
            if player not in self.players_data:
                continue
                
            player_info = self.players_data[player]
            df = player_info['data']
            
            # Check if required columns exist
            if all(col in df.columns for col in ['x', 'y', 'timestamp']):
                # Filter data to the trail time window
                mask = (df['timestamp'] >= trail_start_time) & (df['timestamp'] <= current_time)
                trail_df = df[mask]
                
                if not trail_df.empty:
                    # Create trail data
                    trail_data[player] = {
                        'x': trail_df['x'].tolist(),
                        'y': trail_df['y'].tolist(),
                        'timestamps': trail_df['timestamp'].tolist(),
                        'color': player_info['color'],
                        'id': player_info['id']
                    }
        
        return trail_data

def diagnose_player_data_coverage(self, player_key=None):
    """Diagnose actual data coverage for players
    
    Args:
        player_key (str, optional): Specific player to check
        
    Returns:
        dict: Information about player data coverage
    """
    players_to_check = [player_key] if player_key else self.players_data.keys()
    results = {}
    
    for player in players_to_check:
        if player not in self.players_data:
            results[player] = {"error": "Player not found"}
            continue
            
        player_info = self.players_data[player]
        df = player_info['data']
        
        if df.empty:
            results[player] = {"error": "No data available"}
            continue
        
        # Get timestamp ranges
        timestamps = sorted(df['timestamp'].dropna().tolist())
        if not timestamps:
            results[player] = {"error": "No valid timestamps"}
            continue
            
        # Split into hour buckets to check coverage
        hour_buckets = {}
        for ts in timestamps:
            hour_key = ts.strftime('%Y-%m-%d %H:00')
            if hour_key not in hour_buckets:
                hour_buckets[hour_key] = []
            hour_buckets[hour_key].append(ts)
        
        # Summarize each hour bucket
        bucket_stats = {}
        for hour, times in hour_buckets.items():
            bucket_stats[hour] = {
                "count": len(times),
                "min": min(times),
                "max": max(times),
                "duration_minutes": (max(times) - min(times)).total_seconds() / 60
            }
        
        # Get overall stats
        results[player] = {
            "id": player_info['id'],
            "total_rows": len(df),
            "first_timestamp": min(timestamps),
            "last_timestamp": max(timestamps),
            "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600,
            "hour_coverage": bucket_stats
        }
    
    return results

# Function to create visualizations for the current time
def create_visualization(player_manager, current_time, trail_duration, time_threshold=5, force_display_all=True):
    """Create visualization with proper handling of multi-session players
    
    Args:
        player_manager (PlayerDataManager): Player data manager
        current_time (datetime): Current time for visualization
        trail_duration (int): Duration of trail in seconds
        time_threshold (int): Maximum time difference to consider a player active
        force_display_all (bool): Force display of all players
        
    Returns:
        go.Figure: Plotly figure with player visualization
    """
    # Create pitch layout
    fig = create_pitch_layout()
    
    # Get current player positions with specified parameters
    positions = player_manager.get_current_positions(
        current_time, 
        time_threshold, 
        force_display_all=force_display_all
    )
    
    # Add timestamp annotation
    fig.add_annotation(
        x=50, y=105,
        text=f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )
    
    # Track multi-session players
    multi_session_players = []
    
    # Categorize players
    active_players = {}
    inactive_players = {}
    
    for player, pos in positions.items():
        if pos.get('multi_session_player', False):
            multi_session_players.append(player)
            
        if pos.get('active', False):
            active_players[player] = pos
        else:
            inactive_players[player] = pos
    
    # Get player trail positions for active players
    active_player_keys = list(active_players.keys())
    trail_data = {}
    if active_player_keys:  # Only proceed if there are active players
        trail_data = player_manager.get_trail_positions(
            current_time, 
            trail_duration, 
            specific_players=active_player_keys
        )
    
    # Add trails for active players
    for player, trail in trail_data.items():
        fig.add_trace(
            go.Scatter(
                x=trail['x'],
                y=trail['y'],
                mode='lines',
                line=dict(color=trail['color'], width=2, dash='dot'),
                opacity=0.8,
                name=player,
                hoverinfo='none',
                showlegend=True
            )
        )
    
    # Add active players
    for player, pos in active_players.items():
        # Special marker for multi-session players
        is_multi = player in multi_session_players
        marker_symbol = 'star' if is_multi else 'circle'
        
        # Add player marker
        fig.add_trace(
            go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='markers',
                marker=dict(
                    size=12 if is_multi else 10,
                    symbol=marker_symbol,
                    color=pos['color'],
                    line=dict(color='white', width=1)
                ),
                name=player,
                hoverinfo='text',
                hovertext=(f"{player}<br>Multi-session player<br>" if is_multi else f"{player}<br>") + 
                         f"Session: {pos.get('session_id', 'Unknown')}<br>" + 
                         f"Speed: {pos['speed']:.2f} m/s<br>" + 
                         f"Time diff: {pos.get('time_diff', 0):.1f}s",
                showlegend=False
            )
        )
        
        # Add player number label
        fig.add_trace(
            go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='text',
                text=[pos['id']],
                textposition='middle center',
                opacity=1.0,
                textfont=dict(
                    color='white',
                    size=10,
                    family='Arial, sans-serif'
                ),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Add inactive players with different style
    for player, pos in inactive_players.items():
        # Special marker for multi-session players
        is_multi = player in multi_session_players
        marker_symbol = 'star-open' if is_multi else 'circle-open'
        
        # Add player marker
        fig.add_trace(
            go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='markers',
                marker=dict(
                    size=10 if is_multi else 8,
                    symbol=marker_symbol,
                    color=pos['color'],
                    opacity=0.6,
                    line=dict(color='white', width=1)
                ),
                name=player,
                hoverinfo='text',
                hovertext=(f"{player}<br>Multi-session player<br>" if is_multi else f"{player}<br>") +
                         f"NOT ACTIVE AT THIS TIME<br>" +
                         f"Session: {pos.get('session_id', 'Unknown')}<br>" +
                         f"Time diff: {pos.get('time_diff', 0):.1f}s",
                showlegend=False
            )
        )
        
        # Add player number label
        fig.add_trace(
            go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='text',
                text=[pos['id']],
                textposition='middle center',
                opacity=0.6,
                textfont=dict(
                    color='gray',
                    size=9,
                    family='Arial, sans-serif'
                ),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Add legend entries
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle',
                color='white'
            ),
            name=f'Active Player (within {time_threshold}s)',
            showlegend=True
        )
    )
    
    if inactive_players:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=8,
                    symbol='circle-open',
                    color='white',
                    line=dict(color='white', width=1)
                ),
                name='Inactive Player',
                showlegend=True
            )
        )
    
    if multi_session_players:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='star',
                    color='white',
                    line=dict(color='white', width=1)
                ),
                name='Multi-session Player',
                showlegend=True
            )
        )
    
    return fig

def main():
    st.set_page_config(
        page_title="Player GPS Tracking Visualization",
        page_icon="⚽",
        layout="wide"
    )
    
    # App title and description
    st.title("⚽ Player GPS Tracking Visualization")
    st.markdown(
        """
        This app visualizes GPS tracking data of all available football players during a match session.
        Each player's position and movement trail are displayed on a football pitch.
        """
    )
    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Visualization", "Analysis"])
    
    # Set data directory
    default_data_dir = "/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/outputs/fetched_data"
    data_dir = st.sidebar.text_input("Data directory path", value=default_data_dir)
    
    # Initialize player data manager
    if 'player_manager' not in st.session_state:
        st.session_state.player_manager = PlayerDataManager(data_dir)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # Load data when user clicks the button
    if st.sidebar.button("Load Player Data"):
        with st.spinner("Loading player data..."):
            player_manager = st.session_state.player_manager
            player_manager.data_folder = data_dir  # Update data folder if changed
            players_data, global_min_time, global_max_time = player_manager.load_all_player_data()
            
            if not players_data:
                st.error("No valid player data found.")
                return
            
            st.session_state.loaded = True
            st.success(f"Successfully loaded data for {len(players_data)} players!")
            
            # Display unavailable players if any
            if player_manager.unavailable_players:
                st.warning(f"The following players' data could not be loaded: {', '.join(player_manager.unavailable_players)}")
            
            # Create analyzer using loaded player manager
            st.session_state.analyzer = PlayerDataAnalyzer(player_manager=player_manager)
            st.success("Analysis tools ready!")
    
    # If data is not loaded yet, show a message and exit
    if not st.session_state.get('loaded', False) or not st.session_state.player_manager.loaded:
        st.info("Click 'Load Player Data' button in the sidebar to start.")
        return
    
    # Get player data manager
    player_manager = st.session_state.player_manager
    
    # Show different page based on navigation selection
    if page == "Visualization":
        # Animation settings
        st.sidebar.header("Animation Settings")
        trail_duration = st.sidebar.slider("Trail duration (seconds)", 1, 20, 5)
        animation_speed = st.sidebar.slider(
        "Animation Speed", 
        min_value=0.1,  # Allow very slow speeds
        max_value=5.0,  # Allow very fast speeds
        value=1.0,      # Default to normal speed
        step=0.1,       # Allow fine-grained control
        help="Adjust animation speed. Values < 1 slow down, values > 1 speed up.",
        key="animation_speed"  # Add a unique key
    )
        
        # Display options
        st.sidebar.header("Display Options")
        force_display_all = st.sidebar.checkbox(
            "Show all players (even inactive)", 
            value=True,
            help="Force display of all players, even those with no data points near the current time"
        )
        
        # Display all available players
        st.sidebar.header("Players")
        available_players = list(player_manager.players_data.keys())
        st.sidebar.write(f"Visualizing all {len(available_players)} available players")
        
        # List player IDs in the sidebar
        player_ids = [player_manager.players_data[player]['id'] for player in available_players]
        st.sidebar.write("Player IDs: " + ", ".join(player_ids))
        
        # Time control
        global_min_time = player_manager.global_min_time
        global_max_time = player_manager.global_max_time
        
        # Calculate total duration in seconds
        total_duration_seconds = int((global_max_time - global_min_time).total_seconds())
        
        # Time slider step (try to create ~100 steps)
        time_slider_step = max(1, total_duration_seconds // 100)
        
        st.write("### Time Control")
        # Create two columns - one for the slider and one for the jump controls
        time_col1, time_col2 = st.columns([3, 1])

        with time_col1:
            # Make sure last_animation_position is an integer to avoid type errors
            slider_default = int(st.session_state.get('last_animation_position', 0))
            current_time_seconds = st.slider(
                "Match Timeline",
                min_value=0,
                max_value=total_duration_seconds,
                step=time_slider_step,
                value=slider_default,  # Use last animation position as default, ensuring it's an int
                format="%d s",
                key="timeline_slider"
            )

        with time_col2:
            # Add a button to jump to the specified time
            if st.button("Jump to Current Time"):
                # Update the last animation position to the current slider value
                st.session_state.last_animation_position = int(current_time_seconds)
                # Stop any ongoing animation
                st.session_state.stop_animation = True
                st.session_state.animating = False
                # Rerun the app to reflect the changes
                st.rerun()
        
        # Calculate current time
        current_time = global_min_time + timedelta(seconds=current_time_seconds)
        
        # Add a slider to adjust time threshold
        time_threshold = st.slider(
            "Time Threshold (seconds)",
            min_value=1,
            max_value=30,
            value=5,
            help="Maximum time difference in seconds to consider a player's data point valid"
        )
        
        # Get positions with the specified threshold
        positions = player_manager.get_current_positions(
            current_time, 
            time_threshold, 
            force_display_all=force_display_all
        )
        
        visible_players = len(positions)
        total_players = len(player_manager.players_data)
        
        # Count active players
        active_count = sum(1 for pos in positions.values() if pos.get('active', False))
        
        # Display time and player visibility info
        st.write(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Active players: {active_count}/{total_players} | Shown players: {visible_players}/{total_players} | Threshold: {time_threshold}s")
        
        # Create visualization for current time with all players and the specified time threshold
        fig = create_visualization(
            player_manager, 
            current_time, 
            trail_duration, 
            time_threshold,
            force_display_all=force_display_all
        )
        
        # Display the visualization
        st.plotly_chart(fig, use_container_width=True, key="main_chart")
        
        # Show player visibility diagnostics
        if hasattr(player_manager, 'missing_players_info') and player_manager.missing_players_info:
            with st.expander("Player Visibility Diagnostics", expanded=False):
                st.subheader("Missing Players")
                
                if not player_manager.missing_players_info:
                    st.success("All players are currently visible on the plot.")
                else:
                    st.write("The following players are not included in the plot:")
                    
                    # Create a DataFrame for better display
                    missing_data = []
                    for player, info in player_manager.missing_players_info.items():
                        player_id = player_manager.players_data[player]['id']
                        missing_data.append({
                            "Player": player,
                            "ID": player_id,
                            "Reason": info['reason'],
                            "Time Difference (s)": round(info['time_diff'], 2) if info['time_diff'] != float('inf') else "N/A"
                        })
                    
                    # Display as a table
                    if missing_data:
                        missing_df = pd.DataFrame(missing_data)
                        st.dataframe(missing_df)
                        
                        # Show advice
                        st.info("""
                        **Some players are missing from the plot. Possible reasons:**
                        1. The player has no data near the current time 
                        2. The time threshold is too small
                        3. Try enabling "Show all players" to display inactive players
                        """)
        
        # This is the modified section for the animation controls in the main() function
# Replace the animation controls and logic section with this code

    # Animation controls
    st.write("### Animation Controls")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        play_button = st.button("▶️ Play")

    with col2:
        stop_button = st.button("⏹️ Stop")
        
    with col3:
        # Add a button to jump forward 10 seconds
        forward_button = st.button("⏭️ +10s")
        
    with col4:
        # Add a button to jump backward 10 seconds
        backward_button = st.button("⏮️ -10s")

    # Handle forward jump
    # Handle forward jump
    if forward_button:
        # Calculate new position (10 seconds forward)
        current_pos = int(st.session_state.get('last_animation_position', 0))
        new_position = min(total_duration_seconds, current_pos + 10)
        st.session_state.last_animation_position = int(new_position)
        # Stop any ongoing animation
        st.session_state.stop_animation = True
        st.session_state.animating = False
        # Rerun the app to reflect the changes
        st.rerun()
        
    # Handle backward jump
    if backward_button:
        # Calculate new position (10 seconds backward)
        current_pos = int(st.session_state.get('last_animation_position', 0))
        new_position = max(0, current_pos - 10)
        st.session_state.last_animation_position = int(new_position)
        # Stop any ongoing animation
        st.session_state.stop_animation = True
        st.session_state.animating = False
        # Rerun the app to reflect the changes
        st.rerun()
    # Store the last animation position in session state
    if 'last_animation_position' not in st.session_state:
        st.session_state.last_animation_position = current_time_seconds

    # Animation logic
    if play_button:
        # Reset the stop_animation flag
        st.session_state.stop_animation = False
        st.session_state.animating = True
        
        # Create a placeholder for the animated plot
        animated_plot = st.empty()
        
        # Create a placeholder for the time indicator
        time_indicator = st.empty()
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        
        # Animation starting point - either from last position or current slider position
        start_frame = st.session_state.last_animation_position if st.session_state.get('animating', False) else current_time_seconds
        
        # Animation loop
        frame_counter = 0
        progress = start_frame
        
        while progress <= total_duration_seconds:
            # Check if animation should stop
            if st.session_state.get('stop_animation', False):
                # Save current position for later resuming
                st.session_state.last_animation_position = progress
                break
            
            # Calculate current animation time
            anim_time = global_min_time + timedelta(seconds=progress)
            
            # Calculate elapsed time from start of animation
            elapsed_seconds = int(progress)  # Convert to integer for whole seconds
            
            # Update progress bar - calculate percentage
            progress_percentage = min(1.0, progress / total_duration_seconds)
            progress_bar.progress(progress_percentage)
            
            # Create new visualization frame with all players and the current time threshold
            anim_fig = create_visualization(
                player_manager,
                anim_time,
                trail_duration,
                time_threshold,
                force_display_all=force_display_all
            )
            
            # Get current positions for active player count
            anim_positions = player_manager.get_current_positions(
                anim_time,
                time_threshold,
                force_display_all=force_display_all
            )
            
            # Count active players
            anim_active_count = sum(1 for pos in anim_positions.values() if pos.get('active', False))
            visible_count = len(anim_positions)
            
            # Show player count and timing information
            time_indicator.write(
                f"Current time: {anim_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Elapsed Time: {elapsed_seconds} s | "
                f"Active players: {anim_active_count}/{total_players} | "
                f"Shown players: {visible_count}/{total_players} | "
                f"Threshold: {time_threshold}s"
            )
            
            # Display the updated visualization with a unique key for each frame
            animated_plot.plotly_chart(anim_fig, use_container_width=True, key=f"animation_frame_{frame_counter}")
            
            # Add a small delay to control animation speed (adjusted by speed setting)
            time.sleep(0.1 / animation_speed)
            
            # Increment progress and frame counter
            progress += animation_speed
            frame_counter += 1
            
            # Update last animation position for potential resuming
            st.session_state.last_animation_position = progress
        
        # If we reached the end, reset position to the beginning for next play
        if progress > total_duration_seconds:
            st.session_state.last_animation_position = 0
            
        # Animation complete or stopped, update state
        st.session_state.animating = False

    # Handle stop button
    if stop_button:
        st.session_state.stop_animation = True    
    elif page == "Analysis":
        # Analysis page
        st.title("Player Data Analysis")
        
        if not st.session_state.get('loaded', False) or st.session_state.analyzer is None:
            st.info("Please load player data first.")
        else:
            # Analysis options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Data Availability", "Time Gaps", "Full Analysis"]
            )
            
            if analysis_type == "Data Availability":
                bin_size = st.slider("Time Bin Size (seconds)", 1, 30, 5)
                
                if st.button("Run Analysis"):
                    with st.spinner("Analyzing..."):
                        fig, ax, results_df = st.session_state.analyzer.analyze_data_availability(
                            bin_size_seconds=bin_size
                        )
                        
                        # Display results
                        st.subheader("Player Data Availability Over Time")
                        st.pyplot(fig)
                        
                        st.subheader("Player Data Availability Statistics")
                        st.dataframe(results_df)
                        
                        # Offer download of statistics
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Statistics CSV",
                            data=csv,
                            file_name="player_availability_stats.csv",
                            mime="text/csv"
                        )
            
            elif analysis_type == "Time Gaps":
                if st.button("Generate Gaps Report"):
                    with st.spinner("Generating report..."):
                        # Use a temporary file
                        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                            # Generate the report
                            st.session_state.analyzer.create_data_gaps_report(tmp.name)
                            
                            # Read the file
                            with open(tmp.name, 'r') as f:
                                report_content = f.read()
                            
                            # Display the report
                            st.subheader("Player Data Gaps Report")
                            st.text_area("Report", report_content, height=500)
                            
                            # Offer download
                            st.download_button(
                                label="Download Full Report",
                                data=report_content,
                                file_name="player_gaps_report.txt",
                                mime="text/plain"
                            )
            
            elif analysis_type == "Full Analysis":
                bin_size = st.slider("Time Bin Size (seconds)", 1, 30, 5)
                output_prefix = st.text_input("Output File Prefix", "player_analysis")
                
                if st.button("Run Full Analysis"):
                    with st.spinner("Running full analysis..."):
                        # Use a temporary directory
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            output_prefix_path = os.path.join(tmpdirname, output_prefix)
                            
                            # Run the analysis
                            fig, ax, results_df = st.session_state.analyzer.run_full_analysis(
                                output_prefix=output_prefix_path,
                                bin_size_seconds=bin_size
                            )
                            
                            # Display visualization
                            st.subheader("Player Data Availability Over Time")
                            st.pyplot(fig)
                            
                            # Display statistics
                            st.subheader("Player Data Availability Statistics")
                            st.dataframe(results_df)
                            
                            # Read the gaps report
                            gaps_report_file = f"{output_prefix_path}_gaps_report.txt"
                            with open(gaps_report_file, 'r') as f:
                                report_content = f.read()
                            
                            # Display the report
                            st.subheader("Player Data Gaps Report")
                            st.text_area("Report", report_content, height=300)
                            
                            # Offer downloads
                            visualization_file = f"{output_prefix_path}_availability.png"
                            with open(visualization_file, 'rb') as f:
                                visualization_bytes = f.read()
                            
                            st.download_button(
                                label="Download Visualization",
                                data=visualization_bytes,
                                file_name=f"{output_prefix}_availability.png",
                                mime="image/png"
                            )
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Statistics CSV",
                                data=csv,
                                file_name=f"{output_prefix}_stats.csv",
                                mime="text/csv"
                            )
                            
                            st.download_button(
                                label="Download Gaps Report",
                                data=report_content,
                                file_name=f"{output_prefix}_gaps_report.txt",
                                mime="text/plain"
                            )

# Run the app
if __name__ == "__main__":
    # Initialize session state
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    
    if 'animating' not in st.session_state:
        st.session_state.animating = False
    
    if 'stop_animation' not in st.session_state:
        st.session_state.stop_animation = False
    
    main()







#streamlit run streamlit-animated-player-tracking.py 
#cd /Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/src/    