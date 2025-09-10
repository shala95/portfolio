import numpy as np
try:
    import nolds  # For entropy calculation (optional)
except ImportError:
    pass
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re
import os
import glob
import traceback
from datetime import datetime, timedelta
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from stretch_index_module import StretchIndexAnalyzer



# Set up page configuration
st.set_page_config(
    page_title="Team Formation & Distance Analysis",
    page_icon="⚽",
    layout="wide"
)

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

        # Additional colors for positional roles
        self.goalkeeper_color = self.bright_blue
        self.defender_color = self.steel_blue
        self.midfielder_color = self.medium_green
        self.forward_color = self.sea_green
        self.winger_color = self.sky_blue

# Function to create a bar chart of distance metrics
def create_distance_metrics_chart(metrics, metric_type='avg_distance', title="Distance Metrics"):
    """Create a bar chart showing distance metrics by team line"""
    data = []
    labels = []
    
    for line_name, line_data in metrics.items():
        if line_name in ['line_distances', 'team']:
            continue
            
        if metric_type in line_data:
            data.append(line_data[metric_type])
            labels.append(line_name)
    
    # Create the figure
    fig = go.Figure()
    palette = SoccermentPalette()
    
    # Define colors for different position categories
    position_colors = {
        'Goalkeeper': palette.goalkeeper_color,
        'Defender': palette.defender_color,
        'Midfielder': palette.midfielder_color,
        'Forward': palette.forward_color,
        'Winger': palette.winger_color
    }
    
    colors = [position_colors.get(label, palette.light_grey) for label in labels]
    
    # Add the bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=data,
            marker_color=colors,
            text=[f"{v:.2f}" for v in data],
            textposition="auto"
        )
    )
    
    # Determine y-axis label based on metric type
    y_label = "Distance (m)"
    if metric_type == 'x_spread':
        y_label = "Horizontal Spread (m)"
    elif metric_type == 'y_spread':
        y_label = "Vertical Spread (m)"
    elif metric_type == 'avg_horizontal_distance':
        y_label = "Avg. Horizontal Distance (m)"
    elif metric_type == 'avg_vertical_distance':
        y_label = "Avg. Vertical Distance (m)"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Team Line",
        yaxis_title=y_label,
        template="plotly_dark",
        height=400,
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        font=dict(color=palette.off_white)
    )
    
    return fig

# Function to create heatmap of player-to-player distances
def create_player_distance_heatmap(player_distances, title="Player-to-Player Distances"):
    """Create a heatmap visualization of player-to-player distances"""
    # Extract unique players while preserving order
    players = []
    for dist in player_distances:
        if dist['player1'] not in players:
            players.append(dist['player1'])
        if dist['player2'] not in players:
            players.append(dist['player2'])
    
    # Create empty matrix for distances
    n = len(players)
    distance_matrix = np.zeros((n, n))
    
    # Fill in the distance matrix
    for dist in player_distances:
        i = players.index(dist['player1'])
        j = players.index(dist['player2'])
        distance_matrix[i, j] = dist['distance']
        distance_matrix[j, i] = dist['distance']  # Mirror the matrix
    
    # Create the heatmap figure
    fig = go.Figure()
    palette = SoccermentPalette()
    
    # Position labels with role information
    labels = []
    for player in players:
        for dist in player_distances:
            if dist['player1'] == player:
                position = dist['player1_position']
                break
            elif dist['player2'] == player:
                position = dist['player2_position']
                break
        else:
            position = "Unknown"
        
        # Create label with position
        short_name = player.split()[-1] if len(player.split()) > 1 else player
        labels.append(f"{short_name} ({position})")
    
    # Add the heatmap
    fig.add_trace(
        go.Heatmap(
            z=distance_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            text=[[f"{distance_matrix[i][j]:.2f}m" if i != j else "" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
            colorbar=dict(title="Distance (m)")
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=700,
        width=700,
        template="plotly_dark",
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        font=dict(color=palette.off_white),
        xaxis=dict(tickangle=45)
    )
    
    return fig

# Function to create a line chart of metrics over time periods
def create_metrics_time_series(period_metrics, metric_name='avg_distance', line_name='Defender'):
    """Create a line chart of metrics over time periods"""
    periods = []
    values = []
    
    for period_key, metrics in period_metrics.items():
        # Skip intervals and focus on match halves for clarity
        if period_key.startswith('interval_'):
            continue
            
        if line_name in metrics and metric_name in metrics[line_name]:
            periods.append(period_key)
            values.append(metrics[line_name][metric_name])
    
    # Create figure
    fig = go.Figure()
    palette = SoccermentPalette()
    
    # Add the line chart
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=values,
            mode='lines+markers',
            line=dict(width=3, color=palette.bright_blue),
            marker=dict(size=10)
        )
    )
    
    # Format title based on metric name
    metric_display = metric_name.replace('_', ' ').title()
    
    # Update layout
    fig.update_layout(
        title=f"{metric_display} for {line_name}s Over Time",
        xaxis_title="Match Period",
        yaxis_title=f"{metric_display} (m)",
        template="plotly_dark",
        height=400,
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        font=dict(color=palette.off_white)
    )
    
    return fig

# Function to create a radar chart of team metrics
def create_team_radar_chart(metrics):
    """Create a radar chart of team metrics"""
    # Extract team line metrics
    categories = []
    values = []
    
    for line_name, line_data in metrics.items():
        if line_name in ['line_distances', 'team']:
            continue
            
        if 'avg_distance' in line_data:
            categories.append(f"{line_name} Dist")
            values.append(line_data['avg_distance'])
            
        if 'x_spread' in line_data:
            categories.append(f"{line_name} Width")
            values.append(line_data['x_spread'])
            
        if 'y_spread' in line_data:
            categories.append(f"{line_name} Depth")
            values.append(line_data['y_spread'])
    
    # Add team-level metrics
    if 'team' in metrics:
        team_data = metrics['team']
        if 'x_spread' in team_data:
            categories.append("Team Width")
            values.append(team_data['x_spread'])
        if 'y_spread' in team_data:
            categories.append("Team Depth")
            values.append(team_data['y_spread'])
    
    # Create radar chart
    fig = go.Figure()
    palette = SoccermentPalette()
    
    # Add the radar chart - with fixed transparency format
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color=palette.bright_blue),
            # Use rgba format for transparency instead of hex + alpha
            fillcolor=f'rgba(28, 130, 190, 0.25)'  # This is equivalent to #1C82BE with 25% opacity
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Team Formation Metrics",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2 if values else 1]  # Add some headroom and handle empty values
            )
        ),
        template="plotly_dark",
        height=500,
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        font=dict(color=palette.off_white)
    )
    
    return fig

# Function to calculate and display line-to-line distances
def create_line_distance_chart(metrics):
    """Create a chart showing distances between team lines"""
    if 'line_distances' not in metrics:
        return None
        
    line_pairs = []
    distances = []
    horizontal_distances = []
    vertical_distances = []
    
    for pair in metrics['line_distances']:
        line_pairs.append(f"{pair['line1']}-{pair['line2']}")
        distances.append(pair['distance'])
        horizontal_distances.append(pair['horizontal_distance'])
        vertical_distances.append(pair['vertical_distance'])
    
    # Create figure
    fig = go.Figure()
    palette = SoccermentPalette()
    
    # Add the traces
    fig.add_trace(
        go.Bar(
            x=line_pairs,
            y=distances,
            name="Total Distance",
            marker_color=palette.bright_blue
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=line_pairs,
            y=horizontal_distances,
            name="Horizontal Distance",
            marker_color=palette.steel_blue
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=line_pairs,
            y=vertical_distances,
            name="Vertical Distance",
            marker_color=palette.medium_green
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Distances Between Team Lines",
        xaxis_title="Line Pairs",
        yaxis_title="Distance (m)",
        template="plotly_dark",
        height=500,
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        font=dict(color=palette.off_white),
        barmode='group'
    )
    
    return fig

# Function to export metrics to CSV
def export_metrics_to_csv(metrics, time_period_key):
    """Export metrics to CSV format"""
    # Prepare data for CSV
    data = []
    
    # First, line metrics
    for line_name, line_data in metrics.items():
        if line_name in ['line_distances', 'team']:
            continue
            
        row = {
            'time_period': time_period_key,
            'metric_type': 'line',
            'line_name': line_name,
            'player_count': line_data.get('player_count', 0),
            'avg_distance': line_data.get('avg_distance', 0),
            'avg_horizontal_distance': line_data.get('avg_horizontal_distance', 0),
            'avg_vertical_distance': line_data.get('avg_vertical_distance', 0),
            'x_spread': line_data.get('x_spread', 0),
            'y_spread': line_data.get('y_spread', 0)
        }
        data.append(row)
    
    # Add line-to-line distances
    if 'line_distances' in metrics:
        for pair in metrics['line_distances']:
            row = {
                'time_period': time_period_key,
                'metric_type': 'line_pair',
                'line_name': f"{pair['line1']}-{pair['line2']}",
                'line1': pair['line1'],
                'line2': pair['line2'],
                'distance': pair['distance'],
                'horizontal_distance': pair['horizontal_distance'],
                'vertical_distance': pair['vertical_distance']
            }
            data.append(row)
    
    # Add team metrics
    if 'team' in metrics:
        row = {
            'time_period': time_period_key,
            'metric_type': 'team',
            'line_name': 'Team',
            'player_count': metrics['team'].get('player_count', 0),
            'x_spread': metrics['team'].get('x_spread', 0),
            'y_spread': metrics['team'].get('y_spread', 0),
            'area': metrics['team'].get('area', 0)
        }
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

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

# Class for managing team formation and player tracking data
class TeamDataManager:
    """Class for managing player tracking and team formation data"""
    def __init__(self, tracking_data_path=None, formation_data_path=None):
        self.tracking_data_path = tracking_data_path
        self.formation_data_path = formation_data_path
        self.tracking_data = None
        self.formation_data = None
        self.players = []
        self.positions = {}
        self.global_min_time = None
        self.global_max_time = None
        self.time_periods = {}
        self.team_lines = {} # Store players by positional lines (defense, midfield, attack)
        
    def load_formation_data(self):
        """Load team formation CSV file"""
        try:
            if not self.formation_data_path or not Path(self.formation_data_path).exists():
                st.error(f"Formation data file not found: {self.formation_data_path}")
                return False
                
            # Use pandas to load the CSV for display purposes
            try:
                self.formation_data = pd.read_csv(self.formation_data_path, encoding='latin-1')
            except Exception as e:
                st.warning(f"Warning: Could not load with standard method: {str(e)}")
                try:
                    self.formation_data = pd.read_csv(self.formation_data_path, encoding='latin-1', sep=';')
                except Exception as e2:
                    st.warning(f"Also failed with semicolon separator: {str(e2)}")
                    
            # Direct processing of the CSV file
            players_by_line = {
                'Goalkeeper': [],
                'Defender': [],
                'Midfielder': [],
                'Forward': [],
            }
            
            # Define position categories for matching
            position_categories = {
                'Goalkeeper': ['GK', 'G'],
                'Defender': ['LB', 'CB', 'RB', 'LWB', 'RWB', 'DEF', 'D'],
                'Midfielder': ['CDM', 'CM', 'CAM', 'MID', 'M', 'LW', 'RW', 'LM', 'RM', 'W'],
                'Forward': ['ST', 'CF', 'FW', 'F'],
            }
            
            # Direct file reading to bypass pandas issues
            with open(self.formation_data_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
                
                # Skip header
                for line in lines[1:]:
                    # Remove quotes if present
                    cleaned_line = line.replace('"', '')
                    
                    # Split by comma or semicolon
                    if ',' in cleaned_line:
                        parts = cleaned_line.strip().split(',')
                    else:
                        parts = cleaned_line.strip().split(';')
                    
                    # Check if we have enough parts
                    if len(parts) < 7:
                        continue
                    
                    # Extract player info
                    player_name = parts[4].strip() if parts[4].strip() else f"{parts[2].strip()} {parts[3].strip()}"
                    position = parts[6].strip()
                    shirt_number = parts[5].strip()
                    module_position = 0
                    if len(parts) >= 9 and parts[8].strip():
                        try:
                            module_position = float(parts[8].strip())
                        except ValueError:
                            pass
                    
                    # Determine position category
                    position_category = 'Midfielder'  # Default
                    for category, pos_codes in position_categories.items():
                        if any(position.startswith(code) for code in pos_codes):
                            position_category = category
                            break
                    
                    # Add player to the correct line
                    players_by_line[position_category].append({
                        'player_name': player_name,
                        'position': position,
                        'module_position': module_position,
                        'shirt_number': shirt_number
                    })
            
            # Print debug info
            for category, players in players_by_line.items():
                player_names = [p['player_name'] for p in players]
                st.info(f"{category}: {player_names}")
            
            self.team_lines = players_by_line
            return True
        
        except Exception as e:
            st.error(f"Error loading formation data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
        
    def load_tracking_data(self):
        """Load player tracking data CSV file"""
        try:
            if not self.tracking_data_path or not Path(self.tracking_data_path).exists():
                st.error(f"Tracking data file not found: {self.tracking_data_path}")
                return False
                    
            # Load with low_memory=False to avoid DtypeWarning
            self.tracking_data = pd.read_csv(self.tracking_data_path, low_memory=False)
            
            # Clean column names
            self.tracking_data.columns = [col.strip('"').strip() for col in self.tracking_data.columns]
            
            # Display the available columns for debugging
            st.info(f"Available columns in tracking data: {', '.join(self.tracking_data.columns)}")
            
            # Look for local_time column with flexible names
            local_time_columns = ['local_time', 'timestamp', 'time', 'date_time', 'gps_time']
            found_local_time_col = None
            
            for col in local_time_columns:
                if col in self.tracking_data.columns:
                    found_local_time_col = col
                    break
            
            if found_local_time_col:
                st.success(f"Using '{found_local_time_col}' as local_time column")
                
                # Convert local_time column to datetime using mixed format
                try:
                    self.tracking_data['local_time'] = pd.to_datetime(
                        self.tracking_data[found_local_time_col], 
                        format='mixed',
                        errors='coerce'
                    )
                    
                    # Check for any invalid datetime values
                    invalid_times = self.tracking_data['local_time'].isna().sum()
                    if invalid_times > 0:
                        st.warning(f"Dropped {invalid_times} rows with invalid local_time")
                        self.tracking_data = self.tracking_data.dropna(subset=['local_time'])
                        
                    # Verify that local_time is now a datetime column
                    if pd.api.types.is_datetime64_dtype(self.tracking_data['local_time']):
                        st.success(f"Successfully converted local_time to datetime format")
                    else:
                        st.warning(f"local_time column is not in datetime format, current type: {self.tracking_data['local_time'].dtype}")
                
                except Exception as dt_error:
                    st.error(f"Error converting local_time to datetime: {str(dt_error)}")
                    return False
                
                # Get global time range 
                self.global_min_time = self.tracking_data['local_time'].min()
                self.global_max_time = self.tracking_data['local_time'].max()
                
                # Extract unique players
                if 'player_name' in self.tracking_data.columns:
                    self.players = self.tracking_data['player_name'].unique().tolist()
                else:
                    st.warning("Could not find player_name column")
                    self.players = []
                
                # Identify time periods (halves, 5-minute intervals)
                self.identify_time_periods()
                
                st.success(f"Successfully loaded tracking data for {len(self.players)} players")
                st.info(f"Data timeframe: {self.global_min_time} to {self.global_max_time}")
                return True
            else:
                # Show the actual cleaned column names
                available_cols = ", ".join(self.tracking_data.columns)
                st.error(f"No local_time column found. Available columns: {available_cols}")
                st.info("Please ensure your data has one of these columns: local_time, timestamp, time, date_time, or gps_time")
                return False
                    
        except Exception as e:
            st.error(f"Error loading tracking data: {str(e)}")
            st.code(traceback.format_exc())  # Show the full traceback for better debugging
            return False
    def identify_time_periods(self):
        """Identify and segment the data into halves and smaller time intervals"""
        if self.tracking_data is None or self.global_min_time is None:
            return
            
        # Calculate match duration in minutes
        match_duration = (self.global_max_time - self.global_min_time).total_seconds() / 60
        
        # Create basic periods dictionary
        periods = {
            'full_match': {
                'start': self.global_min_time,
                'end': self.global_max_time,
                'duration_minutes': match_duration
            }
        }
        
        # First half is typically 45 minutes
        first_half_end = self.global_min_time + timedelta(minutes=45)
        
        # Second half start is typically after a 15-minute break
        second_half_start = first_half_end + timedelta(minutes=15)
        
        # Add first half if match duration is sufficient
        if match_duration > 5:  # At least some data to analyze
            periods['first_half'] = {
                'start': self.global_min_time,
                'end': first_half_end,
                'duration_minutes': 45
            }
            
        # Add second half if match duration suggests we have second half data
        if match_duration > 55:  # 45 minutes first half + some break + some second half data
            # Find second half end (expected match is 90 minutes total playing time)
            second_half_end = min(second_half_start + timedelta(minutes=45), self.global_max_time)
            
            periods['second_half'] = {
                'start': second_half_start,
                'end': second_half_end,
                'duration_minutes': (second_half_end - second_half_start).total_seconds() / 60
            }
        
        # Create 5-minute intervals across the match
        interval_start = self.global_min_time
        interval_minutes = 5
        interval_count = 1
        
        while interval_start < self.global_max_time:
            interval_end = min(interval_start + timedelta(minutes=interval_minutes), self.global_max_time)
            
            periods[f'interval_{interval_count}'] = {
                'start': interval_start,
                'end': interval_end,
                'duration_minutes': interval_minutes,
                'label': f"{interval_start.strftime('%H:%M:%S')} - {interval_end.strftime('%H:%M:%S')}"
            }
            
            interval_start = interval_end
            interval_count += 1
        
        self.time_periods = periods
    
    def get_player_positions(self, time_period_key='full_match'):
        """Get average player positions for a specific time period"""
        if self.tracking_data is None:
            return {}
            
        if time_period_key not in self.time_periods:
            st.warning(f"Time period '{time_period_key}' not found")
            return {}
            
        time_period = self.time_periods[time_period_key]
        start_time = time_period['start']
        end_time = time_period['end']
        
        # Filter data for the specified time period
        period_data = self.tracking_data[
            (self.tracking_data['local_time'] >= start_time) & 
            (self.tracking_data['local_time'] <= end_time)
        ]
        
        # Calculate average position for each player during this period
        player_positions = {}
        for player in self.players:
            player_data = period_data[period_data['player_name'] == player]
            
            if not player_data.empty:
                avg_x = player_data['x'].mean()
                avg_y = player_data['y'].mean()
                
                # Get player position from formation data if available
                player_info = self.get_player_info(player)
                
                player_positions[player] = {
                    'x': avg_x,
                    'y': avg_y,
                    'position': player_info.get('position', 'Unknown'),
                    'position_category': player_info.get('position_category', 'Unknown'),
                    'shirt_number': player_info.get('shirt_number', ''),
                    'data_points': len(player_data)
                }
        
        return player_positions
    
    def get_player_info(self, player_name):
        """Find player information in formation data based on player name"""
        if self.formation_data is None:
            return {}
            
        player_info = {}
        
        # Search all categories to find the player
        for category, players in self.team_lines.items():
            for player in players:
                # Try to match names allowing for slight differences
                if self.name_similarity(player_name, player['player_name']):
                    return {
                        'position': player['position'],
                        'position_category': category,
                        'module_position': player['module_position'],
                        'shirt_number': player['shirt_number']
                    }
        
        return {'position': 'Unknown', 'position_category': 'Unknown'}
    
    @staticmethod
    def name_similarity(name1, name2):
        """Check if two player names are similar enough to be considered the same player"""
        # Clean and normalize names
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        # Direct match
        if name1 == name2:
            return True
            
        # Check if last name matches (assuming format "First Last")
        name1_parts = name1.split()
        name2_parts = name2.split()
        
        if len(name1_parts) > 0 and len(name2_parts) > 0:
            # Check if last name matches
            if name1_parts[-1] == name2_parts[-1]:
                return True
                
            # Check if first initial and last name match
            if len(name1_parts) > 1 and len(name2_parts) > 1:
                if name1_parts[0][0] == name2_parts[0][0] and name1_parts[-1] == name2_parts[-1]:
                    return True
        
        return False
    
    def calculate_team_lines(self, positions):
        """Group players by position category (defense, midfield, attack)"""
        lines = {
            'Goalkeeper': [],
            'Defender': [],
            'Midfielder': [],
            'Forward': [],
            'Winger': []
        }
        
        for player, data in positions.items():
            category = data['position_category']
            if category in lines:
                lines[category].append({
                    'player': player,
                    'x': data['x'],
                    'y': data['y'],
                    'position': data['position'],
                    'shirt_number': data['shirt_number']
                })
        
        return lines
    
    def calculate_line_metrics(self, positions):
        """Calculate metrics for each team line"""
        lines = self.calculate_team_lines(positions)
        metrics = {}
        
        # Calculate within-line metrics for each line
        for line_name, players in lines.items():
            if len(players) < 2:
                # Need at least 2 players to calculate distances
                continue
                
            # Calculate average position of the line
            x_positions = [p['x'] for p in players]
            y_positions = [p['y'] for p in players]
            
            avg_x = sum(x_positions) / len(x_positions)
            avg_y = sum(y_positions) / len(y_positions)
            
            # Calculate horizontal and vertical spread
            x_spread = max(x_positions) - min(x_positions)
            y_spread = max(y_positions) - min(y_positions)
            
            # Calculate average distances between players in the line
            player_distances = []
            for i in range(len(players)):
                for j in range(i+1, len(players)):
                    p1 = players[i]
                    p2 = players[j]
                    
                    dist = self.calculate_distance(p1['x'], p1['y'], p2['x'], p2['y'])
                    horizontal_dist = abs(p1['x'] - p2['x'])
                    vertical_dist = abs(p1['y'] - p2['y'])
                    
                    player_distances.append({
                        'player1': p1['player'],
                        'player2': p2['player'],
                        'distance': dist,
                        'horizontal_distance': horizontal_dist,
                        'vertical_distance': vertical_dist
                    })
            
            avg_distance = sum(d['distance'] for d in player_distances) / len(player_distances) if player_distances else 0
            avg_horizontal = sum(d['horizontal_distance'] for d in player_distances) / len(player_distances) if player_distances else 0
            avg_vertical = sum(d['vertical_distance'] for d in player_distances) / len(player_distances) if player_distances else 0
            
            metrics[line_name] = {
                'avg_position': (avg_x, avg_y),
                'x_spread': x_spread,
                'y_spread': y_spread,
                'avg_distance': avg_distance,
                'avg_horizontal_distance': avg_horizontal,
                'avg_vertical_distance': avg_vertical,
                'player_count': len(players),
                'players': players,
                'player_distances': player_distances
            }
        
        # Calculate between-line metrics
        line_pairs = []
        line_names = [l for l in lines.keys() if lines[l]]  # Only consider lines with players
        
        for i in range(len(line_names)):
            for j in range(i+1, len(line_names)):
                line1 = line_names[i]
                line2 = line_names[j]
                
                if line1 in metrics and line2 in metrics:
                    # Calculate distance between average positions of the lines
                    pos1 = metrics[line1]['avg_position']
                    pos2 = metrics[line2]['avg_position']
                    
                    distance = self.calculate_distance(pos1[0], pos1[1], pos2[0], pos2[1])
                    horizontal_distance = abs(pos1[0] - pos2[0])
                    vertical_distance = abs(pos1[1] - pos2[1])
                    
                    line_pairs.append({
                        'line1': line1,
                        'line2': line2,
                        'distance': distance,
                        'horizontal_distance': horizontal_distance,
                        'vertical_distance': vertical_distance
                    })
        
        metrics['line_distances'] = line_pairs
        
        # Calculate compactness (total team spread)
        all_players = []
        for players in lines.values():
            all_players.extend(players)
        
        if all_players:
            x_positions = [p['x'] for p in all_players]
            y_positions = [p['y'] for p in all_players]
            
            x_spread = max(x_positions) - min(x_positions)
            y_spread = max(y_positions) - min(y_positions)
            
            metrics['team'] = {
                'x_spread': x_spread,
                'y_spread': y_spread,
                'area': x_spread * y_spread,
                'player_count': len(all_players)
            }
        
        return metrics
    
    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    def calculate_player_to_player_distances(self, positions):
        """Calculate distances between all players"""
        players = list(positions.keys())
        player_distances = []
        
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                player1 = players[i]
                player2 = players[j]
                
                p1_data = positions[player1]
                p2_data = positions[player2]
                
                # Calculate distances
                dist = self.calculate_distance(
                    p1_data['x'], p1_data['y'],
                    p2_data['x'], p2_data['y']
                )
                
                horizontal_dist = abs(p1_data['x'] - p2_data['x'])
                vertical_dist = abs(p1_data['y'] - p2_data['y'])
                
                # Store distance data
                player_distances.append({
                    'player1': player1,
                    'player2': player2,
                    'player1_position': p1_data['position'],
                    'player2_position': p2_data['position'],
                    'player1_category': p1_data['position_category'],
                    'player2_category': p2_data['position_category'],
                    'distance': dist,
                    'horizontal_distance': horizontal_dist,
                    'vertical_distance': vertical_dist
                })
        
        return player_distances
    
    def get_period_distance_metrics(self, period_key='full_match'):
        """Get distance metrics for a specific time period"""
        positions = self.get_player_positions(period_key)
        metrics = self.calculate_line_metrics(positions)
        return metrics, positions
    
    def calculate_distance_over_periods(self):
        """Calculate metrics for all defined time periods"""
        period_metrics = {}
        
        for period_key in self.time_periods.keys():
            metrics, _ = self.get_period_distance_metrics(period_key)
            period_metrics[period_key] = metrics
        
        return period_metrics

# Function to visualize player positions and team structure
def visualize_team_positions(positions, metrics=None, title="Team Positions"):
    """Create visualization of player positions with team structure analysis"""
    fig = create_pitch_layout()
    palette = SoccermentPalette()
    
    # Define colors for different position categories
    position_colors = {
        'Goalkeeper': palette.goalkeeper_color,
        'Defender': palette.defender_color,
        'Midfielder': palette.midfielder_color,
        'Forward': palette.forward_color,
        'Winger': palette.winger_color,
        'Unknown': palette.light_grey
    }
    
    # Add player positions
    for player, data in positions.items():
        # Set color based on position category
        color = position_colors.get(data['position_category'], palette.light_grey)
        
        # Add player marker
        fig.add_trace(
            go.Scatter(
                x=[data['x']],
                y=[data['y']],
                mode='markers',
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                name=player,
                text=f"{player} ({data['position']})",
                hoverinfo='text'
            )
        )
        
        # Add player number if available
        if data.get('shirt_number'):
            fig.add_trace(
                go.Scatter(
                    x=[data['x']],
                    y=[data['y']],
                    mode='text',
                    text=[data['shirt_number']],
                    textposition='middle center',
                    textfont=dict(
                        color='white',
                        size=9
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
            )
    
    # Add team lines and convex hulls if metrics are available
    if metrics:
        # Draw lines between players in the same category
        for line_name, line_data in metrics.items():
            if line_name in ['line_distances', 'team']:
                continue
                
            players = line_data.get('players', [])
            if len(players) < 2:
                continue
                
            # Create line connections between players in the same line
            for i in range(len(players)):
                for j in range(i+1, len(players)):
                    p1 = players[i]
                    p2 = players[j]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[p1['x'], p2['x']],
                            y=[p1['y'], p2['y']],
                            mode='lines',
                            line=dict(
                                color=position_colors.get(line_name, palette.light_grey),
                                width=1,
                                dash='dot'
                            ),
                            showlegend=False,
                            hoverinfo='none'
                        )
                    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        legend=dict(
            title="Players",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Function to create heatmap of player positions
def create_position_heatmap(tracking_data, time_period=None):
    """Create a heatmap of player positions"""
    if tracking_data is None or tracking_data.empty:
        return None
        
    # Filter data for the time period
    if time_period:
        period_data = tracking_data[
            (tracking_data['local_time'] >= time_period['start']) & 
            (tracking_data['local_time'] <= time_period['end'])
        ]
    else:
        period_data = tracking_data
    
    if period_data.empty:
        return None
    
    # Create a 2D histogram of positions
    hist, x_edges, y_edges = np.histogram2d(
        period_data['x'], 
        period_data['y'], 
        bins=[20, 20], 
        range=[[0, 100], [0, 100]]
    )
    
    # Create a heatmap using plotly
    palette = SoccermentPalette()
    
    fig = create_pitch_layout()
    
    # Add the heatmap as a contour
    fig.add_trace(
        go.Heatmap(
            z=hist.T,  # Transpose for correct orientation
            x=[(x_edges[i] + x_edges[i+1])/2 for i in range(len(x_edges)-1)],
            y=[(y_edges[i] + y_edges[i+1])/2 for i in range(len(y_edges)-1)],
            colorscale='Viridis',
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title="Player density",
                thickness=15,
                len=0.5,
                y=0.5,
                yanchor="middle",
                tickvals=[]
            )
        )
    )
    
    # Update layout
    time_range = ""
    if time_period:
        time_range = f" ({time_period['start'].strftime('%H:%M:%S')} - {time_period['end'].strftime('%H:%M:%S')})"
        
    fig.update_layout(
        title=f"Player Position Heatmap{time_range}",
        height=600
    )
    
    return fig

# Function to visualize distance metrics as a network graph
def visualize_distance_network(positions, metrics, title="Player Distance Network"):
    """Create a network graph showing player distances"""
    fig = create_pitch_layout()
    palette = SoccermentPalette()
    
    # Define colors for different position categories
    position_colors = {
        'Goalkeeper': palette.goalkeeper_color,
        'Defender': palette.defender_color,
        'Midfielder': palette.midfielder_color,
        'Forward': palette.forward_color,
        'Winger': palette.winger_color,
        'Unknown': palette.light_grey
    }
    
    # Add player positions
    for player, data in positions.items():
        # Set color based on position category
        color = position_colors.get(data['position_category'], palette.light_grey)
        
        # Add player marker
        fig.add_trace(
            go.Scatter(
                x=[data['x']],
                y=[data['y']],
                mode='markers',
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                name=player,
                text=f"{player} ({data['position']})",
                hoverinfo='text'
            )
        )
        
        # Add player number or initials
        if data.get('shirt_number'):
            text = data['shirt_number']
        else:
            # Use player initials if no shirt number
            parts = player.split()
            text = ''.join([p[0] for p in parts if p])
            
        fig.add_trace(
            go.Scatter(
                x=[data['x']],
                y=[data['y']],
                mode='text',
                text=[text],
                textposition='middle center',
                textfont=dict(
                    color='white',
                    size=9
                ),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Add connections between players with distance info
    for line_name, line_data in metrics.items():
        if line_name in ['line_distances', 'team']:
            continue
            
        # Get player distance data for this line
        player_distances = line_data.get('player_distances', [])
        
        for dist_data in player_distances:
            player1 = dist_data['player1']
            player2 = dist_data['player2']
            
            # Get positions for both players
            p1_pos = positions.get(player1, {})
            p2_pos = positions.get(player2, {})
            
            if not p1_pos or not p2_pos:
                continue
                
            # Calculate line thickness based on distance (closer = thicker line)
            max_thickness = 3
            min_thickness = 0.5
            max_distance = 30  # Maximum expected distance on a soccer field
            
            # Inverse relationship: closer players have thicker lines
            thickness = max(min_thickness, 
                           max_thickness * (1 - min(dist_data['distance'], max_distance) / max_distance))
            
            # Add the connection line
            fig.add_trace(
                go.Scatter(
                    x=[p1_pos['x'], p2_pos['x']],
                    y=[p1_pos['y'], p2_pos['y']],
                    mode='lines',
                    line=dict(
                        color=position_colors.get(line_name, palette.light_grey),
                        width=thickness
                    ),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"{player1} - {player2}: {dist_data['distance']:.2f}m"
                )
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        legend=dict(
            title="Players",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Function to create a
# Main function for the Streamlit app
def main():
    # Set up the title and description
    st.title("⚽ Team Formation & Distance Analysis")
    st.markdown(
        """
        Analyze team formations, player distances, and positional metrics during a match.
        Upload tracking data and team formation information to visualize and calculate key metrics.
        """
    )
    
    # Handle data inputs (file paths or uploads)
    tracking_data_path, formation_data_path = handle_data_inputs()
    
    # Create team data manager
    team_manager = TeamDataManager(tracking_data_path, formation_data_path)
    
    # Add load data button
    if st.sidebar.button("Load Data"):
        load_team_data(team_manager)
    
    # If data is loaded, proceed with analysis
    if st.session_state.get('data_loaded', False):
        run_team_analysis(st.session_state.team_manager)
    else:
        # Show instructions when data is not loaded
        display_help_information()

def handle_data_inputs():
    """Handle file path inputs and file uploads"""
    st.sidebar.header("Data Sources")
    
    # Add file upload option
    st.sidebar.markdown("### Upload Files")
    uploaded_tracking = st.sidebar.file_uploader("Upload Tracking Data CSV", type="csv")
    uploaded_formation = st.sidebar.file_uploader("Upload Formation Data CSV", type="csv")
    
    # Or use file paths
    st.sidebar.markdown("### Or Provide File Paths")
    tracking_data_path = st.sidebar.text_input(
        "Tracking Data CSV Path", 
        "/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/team_stats/XSEED_tracking.csv"
    )
    formation_data_path = st.sidebar.text_input(
        "Team Formation CSV Path", 
        "/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/team_stats/team_formation.csv"
    )
    
    # Add Stretch Index analysis options
    st.sidebar.markdown("### Analysis Options")
    if 'show_stretch_peaks' not in st.session_state:
        st.session_state.show_stretch_peaks = True
    
    st.session_state.show_stretch_peaks = st.sidebar.checkbox(
        "Show Stretch Index Peaks/Dips", 
        value=st.session_state.show_stretch_peaks
    )
    
    # Handle uploaded files
    if uploaded_tracking is not None or uploaded_formation is not None:
        import tempfile
        
        if uploaded_tracking is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_tracking:
                tmp_tracking.write(uploaded_tracking.getvalue())
                tracking_data_path = tmp_tracking.name
        
        if uploaded_formation is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_formation:
                tmp_formation.write(uploaded_formation.getvalue())
                formation_data_path = tmp_formation.name
    
    return tracking_data_path, formation_data_path


def load_team_data(team_manager):
    """Load tracking and formation data"""
    try:
        import traceback
        
        with st.spinner("Loading tracking data..."):
            tracking_success = team_manager.load_tracking_data()
        
        with st.spinner("Loading formation data..."):
            formation_success = team_manager.load_formation_data()
        
        if tracking_success and formation_success:
            st.session_state.data_loaded = True
            st.session_state.team_manager = team_manager
            st.success("Data loaded successfully!")
        else:
            st.error("Error loading data. Check file paths and try again.")
            st.session_state.data_loaded = False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.code(traceback.format_exc())
        st.session_state.data_loaded = False

def run_team_analysis(team_manager):
    """Run the team formation and distance analysis"""
    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    
    # Add time period selection
    time_period_options = list(team_manager.time_periods.keys())
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        time_period_options,
        format_func=lambda x: x.replace('_', ' ').title() if not x.startswith('interval') else 
            team_manager.time_periods[x].get('label', x.replace('_', ' ').title())
    )
    
    # Calculate metrics for the selected period
    metrics, positions = team_manager.get_period_distance_metrics(selected_period)
    
    # Create tabs for different analysis views
    # UPDATED: Added a new tab for Stretch Index
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Team Formation", "Distance Metrics", "Positional Analysis", 
        "Time Series Analysis", "Stretch Index"  # Added new tab
    ])
    
    # Fill each tab with relevant content
    with tab1:
        display_team_formation_tab(team_manager, selected_period, metrics, positions)
    
    with tab2:
        display_distance_metrics_tab(team_manager, selected_period, metrics, positions)
    
    with tab3:
        display_positional_analysis_tab(team_manager, selected_period)
    
    with tab4:
        display_time_series_tab(team_manager, selected_period)
    
    # Add the new tab content
    with tab5:
        display_stretch_index_tab(team_manager, selected_period)

def display_team_formation_tab(team_manager, selected_period, metrics, positions):
    """Display team formation visualization and metrics"""
    st.subheader("Team Formation")
    
    # Display selected time period info
    display_period_info(team_manager, selected_period)
    
    # Create and display team formation visualization
    formation_fig = visualize_team_positions(
        positions, 
        metrics,
        title=f"Team Formation - {selected_period.replace('_', ' ').title()}"
    )
    st.plotly_chart(formation_fig, use_container_width=True)
    
    # Display formation metrics in an expander
    with st.expander("Formation Metrics Details"):
        display_formation_metrics(metrics)
    
    # Create radar chart of team metrics
    st.subheader("Team Formation Radar Chart")
    radar_fig = create_team_radar_chart(metrics)
    st.plotly_chart(radar_fig, use_container_width=True)

def display_distance_metrics_tab(team_manager, selected_period, metrics, positions):
    """Display distance metrics visualizations and tables"""
    st.subheader("Distance Metrics")
    
    # Add metric type selection
    metric_types = {
        'avg_distance': 'Average Distance',
        'avg_horizontal_distance': 'Average Horizontal Distance',
        'avg_vertical_distance': 'Average Vertical Distance',
        'x_spread': 'Horizontal Spread (Width)',
        'y_spread': 'Vertical Spread (Depth)'
    }
    
    selected_metric = st.selectbox(
        "Select Metric Type",
        list(metric_types.keys()),
        format_func=lambda x: metric_types[x]
    )
    
    # Create and display distance metrics chart
    metrics_fig = create_distance_metrics_chart(
        metrics,
        metric_type=selected_metric,
        title=f"{metric_types[selected_metric]} by Team Line"
    )
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Display line-to-line distances
    st.subheader("Line-to-Line Distances")
    line_dist_fig = create_line_distance_chart(metrics)
    if line_dist_fig:
        st.plotly_chart(line_dist_fig, use_container_width=True)
    else:
        st.info("Insufficient data to calculate line-to-line distances.")
    
    # Display a table of all distance metrics
    with st.expander("Detailed Distance Metrics"):
        display_detailed_metrics(metrics, selected_period)
    
    # Show network diagram and heatmap
    display_player_distance_visualizations(team_manager, selected_period, metrics, positions)

def display_positional_analysis_tab(team_manager, selected_period):
    """Display positional analysis visualizations"""
    st.subheader("Positional Analysis")
    
    # Create position heatmap
    heatmap_fig = create_position_heatmap(
        team_manager.tracking_data,
        team_manager.time_periods.get(selected_period)
    )
    
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.warning("Insufficient data to create position heatmap.")
    
    # Display per-player movement analysis
    display_player_movement_analysis(team_manager, selected_period)

def display_time_series_tab(team_manager, selected_period):
    """Display time series analysis of metrics"""
    st.subheader("Metrics Over Time")
    
    # Calculate metrics for all time periods
    period_metrics = team_manager.calculate_distance_over_periods()
    
    # Select line and metric to analyze
    display_time_series_controls_and_chart(team_manager, period_metrics)
    
    # Display time period comparison
    display_interval_analysis(team_manager, period_metrics)
    
    # Display team compactness over time
    display_team_compactness_analysis(period_metrics)

def display_period_info(team_manager, selected_period):
    """Display information about the selected time period"""
    period_info = team_manager.time_periods.get(selected_period, {})
    if period_info:
        st.info(
            f"Time Period: {selected_period.replace('_', ' ').title()}\n"
            f"Start: {period_info['start']}\n"
            f"End: {period_info['end']}\n"
            f"Duration: {period_info['duration_minutes']:.2f} minutes"
        )

def display_formation_metrics(metrics):
    """Display detailed formation metrics for each team line"""
    # Display team lines and their metrics
    for line_name, line_data in metrics.items():
        if line_name in ['line_distances', 'team']:
            continue
            
        st.markdown(f"### {line_name}s")
        st.write(f"Player Count: {line_data['player_count']}")
        st.write(f"Average Position: ({line_data['avg_position'][0]:.2f}, {line_data['avg_position'][1]:.2f})")
        st.write(f"Horizontal Spread (Width): {line_data['x_spread']:.2f}m")
        st.write(f"Vertical Spread (Depth): {line_data['y_spread']:.2f}m")
        st.write(f"Average Player Distance: {line_data['avg_distance']:.2f}m")
        
        # Display players in this line
        st.markdown("**Players:**")
        player_data = []
        for player in line_data['players']:
            player_data.append({
                'Player': player['player'],
                'Position': player['position'],
                'Shirt Number': player['shirt_number'],
                'X Position': f"{player['x']:.2f}",
                'Y Position': f"{player['y']:.2f}"
            })
        
        if player_data:
            st.dataframe(pd.DataFrame(player_data))
        
        st.markdown("---")
    
    # Display team-level metrics
    if 'team' in metrics:
        st.markdown("### Overall Team")
        st.write(f"Total Players: {metrics['team']['player_count']}")
        st.write(f"Team Width (X-Spread): {metrics['team']['x_spread']:.2f}m")
        st.write(f"Team Depth (Y-Spread): {metrics['team']['y_spread']:.2f}m")
        st.write(f"Team Area: {metrics['team']['area']:.2f} sq. meters")

def display_detailed_metrics(metrics, selected_period):
    """Display detailed metrics in a table with download option"""
    # Export metrics to a DataFrame for display
    metrics_df = export_metrics_to_csv(metrics, selected_period)
    st.dataframe(metrics_df)
    
    # Add download button for metrics CSV
    csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="Download Metrics CSV",
        data=csv,
        file_name=f"team_metrics_{selected_period}.csv",
        mime="text/csv"
    )
def create_synchronized_distance_visualizations(team_manager, selected_period):
    """Create data for synchronized visualizations controlled by a time slider"""
    period_info = team_manager.time_periods.get(selected_period, {})
    if not period_info:
        return None, None, None
        
    # Get time range for the period
    start_time = period_info['start']
    end_time = period_info['end']
    
    # Filter tracking data for the period
    period_data = team_manager.tracking_data[
        (team_manager.tracking_data['local_time'] >= start_time) & 
        (team_manager.tracking_data['local_time'] <= end_time)
    ]
    
    # Get unique timestamps (possibly sampling if too many)
    timestamps = period_data['local_time'].unique()
    if len(timestamps) > 50:
        step = len(timestamps) // 50
        timestamps = timestamps[::step]
    
    # Calculate all frames data upfront
    network_frames = []
    heatmap_frames = []
    
    for timestamp in timestamps:
        # Get data for this frame
        frame_data = period_data[period_data['local_time'] == timestamp]
        
        # Calculate player positions for this frame
        positions = {}
        for _, row in frame_data.iterrows():
            player = row['player_name']
            x, y = row['x'], row['y']
            
            # Get player info
            player_info = team_manager.get_player_info(player)
            position_category = player_info.get('position_category', 'Unknown')
            
            positions[player] = {
                'x': x,
                'y': y,
                'position': player_info.get('position', 'Unknown'),
                'position_category': position_category,
                'shirt_number': player_info.get('shirt_number', '')
            }
        
        # Calculate metrics for this frame
        metrics = team_manager.calculate_line_metrics(positions)
        
        # Calculate player-to-player distances
        player_distances = team_manager.calculate_player_to_player_distances(positions)
        
        # Store the network and heatmap data for this frame
        network_frames.append((positions, metrics))
        heatmap_frames.append(player_distances)
    
    return timestamps, network_frames, heatmap_frames
def display_static_distance_visualizations(team_manager, selected_period, metrics, positions):
    """Display static versions of the distance visualizations"""
    # Show network diagram of player distances
    st.subheader("Player Distance Network")
    network_fig = visualize_distance_network(
        positions,
        metrics,
        title=f"Player Distance Network - {selected_period.replace('_', ' ').title()}"
    )
    st.plotly_chart(network_fig, use_container_width=True)
    
    # Calculate player-to-player distances
    player_distances = team_manager.calculate_player_to_player_distances(positions)
    
    # Create and show player distance heatmap
    st.subheader("Player-to-Player Distance Heatmap")
    heatmap_fig = create_player_distance_heatmap(
        player_distances,
        title=f"Player Distance Heatmap - {selected_period.replace('_', ' ').title()}"
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Show player-to-player distance table in an expander
    with st.expander("Detailed Player-to-Player Distances"):
        if player_distances:
            display_player_distance_table(player_distances, selected_period)
        else:
            st.info("No player distance data available.")

def display_player_distance_visualizations(team_manager, selected_period, metrics, positions):
    """Display player distance network and heatmap visualizations"""
    # Add option to animate
    animate_visualizations = st.checkbox("Animate Distance Visualizations", value=False)
    
    if animate_visualizations:
        # Prepare data for all frames
        with st.spinner("Preparing synchronized visualizations..."):
            timestamps, network_frames, heatmap_frames = create_synchronized_distance_visualizations(
                team_manager, selected_period
            )
        
        if timestamps is not None and network_frames and heatmap_frames:
            # Format timestamps for display
            time_options = [ts.strftime('%H:%M:%S') for ts in timestamps]
            
            # Initialize session state for animation control
            if 'playing' not in st.session_state:
                st.session_state.playing = False
                
            if 'frame_idx' not in st.session_state:
                st.session_state.frame_idx = 0
            
            # Add play/pause button
            st.subheader("Timeline Control")
            col1, col2 = st.columns([1, 6])
            
            with col1:
                if st.session_state.playing:
                    if st.button("⏸️ Pause"):
                        st.session_state.playing = False
                else:
                    if st.button("▶️ Play"):
                        st.session_state.playing = True
            
            # Advance frame if playing
            if st.session_state.playing:
                # Increment index
                st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(timestamps)
                # Set autorefresh to continue animation
                st.empty().success("Animation playing... (click Pause to stop)")
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
            
            # Show slider (only affects animation when paused)
            with col2:
                selected_time_idx = st.select_slider(
                    "Select time point:", 
                    options=list(range(len(time_options))),
                    format_func=lambda i: time_options[i],
                    value=st.session_state.frame_idx
                )
            
            # Update the current frame index
            if not st.session_state.playing:
                st.session_state.frame_idx = selected_time_idx
            
            # Get current frame data
            current_positions, current_metrics = network_frames[st.session_state.frame_idx]
            current_player_distances = heatmap_frames[st.session_state.frame_idx]
            
            # Display current time
            st.info(f"Showing data for time: {time_options[st.session_state.frame_idx]}")
            
            # Display both visualizations at the selected time point
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Player Distance Network")
                network_fig = visualize_distance_network(
                    current_positions,
                    current_metrics,
                    title=f"Network at {time_options[st.session_state.frame_idx]}"
                )
                st.plotly_chart(network_fig, use_container_width=True)
            
            with col2:
                st.subheader("Player Distance Heatmap")
                heatmap_fig = create_player_distance_heatmap(
                    current_player_distances,
                    title=f"Heatmap at {time_options[st.session_state.frame_idx]}"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("Could not create synchronized visualizations. Showing static versions instead.")
            display_static_distance_visualizations(team_manager, selected_period, metrics, positions)
    else:
        # Show static visualizations
        display_static_distance_visualizations(team_manager, selected_period, metrics, positions)

def display_player_distance_table(player_distances, selected_period):
    """Display and provide download for player distance table"""
    # Convert to DataFrame for better display
    distances_df = pd.DataFrame(player_distances)
    
    # Reorder and rename columns for clarity
    columns_order = [
        'player1', 'player1_position', 'player1_category',
        'player2', 'player2_position', 'player2_category',
        'distance', 'horizontal_distance', 'vertical_distance'
    ]
    
    column_names = {
        'player1': 'Player 1',
        'player1_position': 'Position 1',
        'player1_category': 'Category 1',
        'player2': 'Player 2',
        'player2_position': 'Position 2',
        'player2_category': 'Category 2',
        'distance': 'Distance (m)',
        'horizontal_distance': 'Horizontal Distance (m)',
        'vertical_distance': 'Vertical Distance (m)'
    }
    
    # Display the table
    show_df = distances_df[columns_order].rename(columns=column_names)
    st.dataframe(show_df)
    
    # Add download button
    csv = show_df.to_csv(index=False)
    st.download_button(
        label="Download Player Distance Data",
        data=csv,
        file_name=f"player_distances_{selected_period}.csv",
        mime="text/csv"
    )

def display_player_movement_analysis(team_manager, selected_period):
    """Display individual player movement analysis"""
    st.subheader("Player Movement Analysis")
    
    # Select player for individual analysis
    selected_player = st.selectbox(
        "Select Player",
        team_manager.players
    )
    
    # Filter data for selected player and time period
    period_info = team_manager.time_periods.get(selected_period, {})
    if period_info and selected_player:
        player_data = team_manager.tracking_data[
            (team_manager.tracking_data['player_name'] == selected_player) &
            (team_manager.tracking_data['local_time'] >= period_info['start']) &
            (team_manager.tracking_data['local_time'] <= period_info['end'])
        ]
        
        if not player_data.empty:
            display_player_metrics(team_manager, player_data, selected_player)
            display_player_trace(player_data, selected_player, selected_period)
            display_player_speed(player_data, selected_player)
        else:
            st.warning(f"No data available for {selected_player} in the selected time period.")

def display_player_metrics(team_manager, player_data, selected_player):
    """Display individual player metrics"""
    # Calculate player movement metrics
    avg_x = player_data['x'].mean()
    avg_y = player_data['y'].mean()
    x_std = player_data['x'].std()
    y_std = player_data['y'].std()
    avg_speed = player_data['speed'].mean() if 'speed' in player_data.columns else 0
    max_speed = player_data['speed'].max() if 'speed' in player_data.columns else 0
    
    # Show player info and metrics
    player_info = team_manager.get_player_info(selected_player)
    st.write(f"**Position:** {player_info.get('position', 'Unknown')}")
    st.write(f"**Position Category:** {player_info.get('position_category', 'Unknown')}")
    
    # Create player metrics columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average X Position", f"{avg_x:.2f}")
        st.metric("X Position Variability", f"{x_std:.2f}")
    
    with col2:
        st.metric("Average Y Position", f"{avg_y:.2f}")
        st.metric("Y Position Variability", f"{y_std:.2f}")
    
    with col3:
        st.metric("Average Speed", f"{avg_speed:.2f} m/s")
        st.metric("Maximum Speed", f"{max_speed:.2f} m/s")

def display_player_trace(player_data, selected_player, selected_period):
    """Display player movement trace visualization"""
    st.subheader(f"{selected_player}'s Movement Trace")
    
    # Create pitch figure
    trace_fig = create_pitch_layout()
    palette = SoccermentPalette()
    
    # Add movement trace
    trace_fig.add_trace(
        go.Scatter(
            x=player_data['x'],
            y=player_data['y'],
            mode='lines',
            line=dict(
                color=palette.bright_blue,
                width=2
            ),
            name=selected_player
        )
    )
    
    # Add average position marker
    avg_x = player_data['x'].mean()
    avg_y = player_data['y'].mean()
    
    trace_fig.add_trace(
        go.Scatter(
            x=[avg_x],
            y=[avg_y],
            mode='markers',
            marker=dict(
                size=15,
                color=palette.medium_green,
                symbol='x',
                line=dict(
                    color='white',
                    width=2
                )
            ),
            name='Average Position'
        )
    )
    
    # Update layout
    trace_fig.update_layout(
        title=f"{selected_player}'s Movement Trace - {selected_period.replace('_', ' ').title()}"
    )
    
    st.plotly_chart(trace_fig, use_container_width=True)

def display_player_speed(player_data, selected_player):
    """Display player speed over time chart"""
    # If speed data is available, show speed over time
    if 'speed' in player_data.columns:
        st.subheader(f"{selected_player}'s Speed Over Time")
        
        # Create a time-aligned version of the local_time
        player_data = player_data.sort_values('local_time')
        player_data['time_elapsed'] = (player_data['local_time'] - player_data['local_time'].min()).dt.total_seconds()
        
        # Create speed chart
        speed_fig = go.Figure()
        palette = SoccermentPalette()
        
        speed_fig.add_trace(
            go.Scatter(
                x=player_data['time_elapsed'],
                y=player_data['speed'],
                mode='lines',
                line=dict(
                    color=palette.medium_blue,
                    width=2
                ),
                name='Speed (m/s)'
            )
        )
        
        # Add reference line for average speed
        avg_speed = player_data['speed'].mean()
        speed_fig.add_trace(
            go.Scatter(
                x=[player_data['time_elapsed'].min(), player_data['time_elapsed'].max()],
                y=[avg_speed, avg_speed],
                mode='lines',
                line=dict(
                    color=palette.medium_green,
                    width=2,
                    dash='dash'
                ),
                name=f"Avg: {avg_speed:.2f} m/s"
            )
        )
        
        # Update layout
        speed_fig.update_layout(
            title=f"{selected_player}'s Speed Profile",
            xaxis_title="Time (seconds)",
            yaxis_title="Speed (m/s)",
            template="plotly_dark",
            height=400,
            paper_bgcolor=palette.charcoal,
            plot_bgcolor=palette.deep_navy,
            font=dict(color=palette.off_white)
        )
        
        st.plotly_chart(speed_fig, use_container_width=True)

def display_time_series_controls_and_chart(team_manager, period_metrics):
    """Display time series chart with controls"""
    # Select line and metric to analyze
    col1, col2 = st.columns(2)
    
    with col1:
        available_lines = []
        for period_key, metrics in period_metrics.items():
            if period_key == 'full_match':  # Use full match to determine available lines
                available_lines = [line for line in metrics.keys() 
                                  if line not in ['line_distances', 'team']]
                break
        
        selected_line = st.selectbox(
            "Select Team Line",
            available_lines,
            index=min(1, len(available_lines)-1) if available_lines else 0  # Default to Defender if available
        )
    
    with col2:
        selected_time_metric = st.selectbox(
            "Select Metric",
            ['avg_distance', 'avg_horizontal_distance', 'avg_vertical_distance', 'x_spread', 'y_spread'],
            format_func=lambda x: {
                'avg_distance': 'Average Distance',
                'avg_horizontal_distance': 'Average Horizontal Distance', 
                'avg_vertical_distance': 'Average Vertical Distance',
                'x_spread': 'Horizontal Spread (Width)',
                'y_spread': 'Vertical Spread (Depth)'
            }[x]
        )
    
    # Create time series chart for selected line and metric
    if available_lines and selected_line and selected_time_metric:
        time_series_fig = create_metrics_time_series(
            period_metrics,
            metric_name=selected_time_metric,
            line_name=selected_line
        )
        st.plotly_chart(time_series_fig, use_container_width=True)
    else:
        st.warning("Insufficient data to create time series analysis.")

def display_interval_analysis(team_manager, period_metrics):
    """Display analysis of 5-minute intervals"""
    st.subheader("5-Minute Interval Analysis")
    
    # Filter for just interval periods
    interval_periods = {k: v for k, v in period_metrics.items() if k.startswith('interval_')}
    
    if interval_periods:
        # Create a comparison chart for intervals
        selected_interval_metric = st.selectbox(
            "Select Metric for Interval Analysis",
            ['avg_distance', 'avg_horizontal_distance', 'avg_vertical_distance', 'x_spread', 'y_spread'],
            format_func=lambda x: {
                'avg_distance': 'Average Distance',
                'avg_horizontal_distance': 'Average Horizontal Distance', 
                'avg_vertical_distance': 'Average Vertical Distance',
                'x_spread': 'Horizontal Spread (Width)',
                'y_spread': 'Vertical Spread (Depth)'
            }[x],
            key="interval_metric_selector"  # Different key from previous selectbox
        )
        
        # Get selected line from earlier selection
        available_lines = []
        for period_key, metrics in period_metrics.items():
            if period_key == 'full_match':  # Use full match to determine available lines
                available_lines = [line for line in metrics.keys() 
                                 if line not in ['line_distances', 'team']]
                break
                
        if available_lines:
            selected_line = available_lines[min(1, len(available_lines)-1)]
            
            # Create data for interval comparison
            interval_labels = []
            interval_values = []
            
            for period_key, metrics in interval_periods.items():
                if selected_line in metrics and selected_interval_metric in metrics[selected_line]:
                    # Get interval label from time periods if available
                    if period_key in team_manager.time_periods:
                        label = team_manager.time_periods[period_key].get('label', period_key)
                    else:
                        label = period_key.replace('_', ' ').title()
                        
                    interval_labels.append(label)
                    interval_values.append(metrics[selected_line][selected_interval_metric])
            
            # Create interval comparison chart
            if interval_labels and interval_values:
                display_interval_chart(interval_labels, interval_values, selected_line, selected_interval_metric)
                
                # Offer full interval data for download
                with st.expander("View All Interval Data"):
                    display_interval_data_table(team_manager, interval_periods)
            else:
                st.info("Not enough interval data to create comparison chart.")
    else:
        st.info("No interval data available for analysis.")

def display_interval_chart(interval_labels, interval_values, selected_line, selected_interval_metric):
    """Display chart of metrics across time intervals"""
    interval_fig = go.Figure()
    palette = SoccermentPalette()
    
    interval_fig.add_trace(
        go.Bar(
            x=interval_labels,
            y=interval_values,
            marker_color=palette.bright_blue,
            text=[f"{v:.2f}" for v in interval_values],
            textposition="auto"
        )
    )
    
    # Format title based on metric name
    metric_display = selected_interval_metric.replace('_', ' ').title()
    
    # Update layout
    interval_fig.update_layout(
        title=f"{metric_display} for {selected_line}s - 5-Minute Intervals",
        xaxis_title="Time Interval",
        yaxis_title=f"{metric_display} (m)",
        template="plotly_dark",
        height=500,
        paper_bgcolor=palette.charcoal,
        plot_bgcolor=palette.deep_navy,
        font=dict(color=palette.off_white)
    )
    
    st.plotly_chart(interval_fig, use_container_width=True)

def display_interval_data_table(team_manager, interval_periods):
    """Display table of all interval data with download option"""
    # Create DataFrame with all interval data
    interval_data = []
    
    for period_key, metrics in interval_periods.items():
        if period_key in team_manager.time_periods:
            time_info = team_manager.time_periods[period_key]
            row = {
                'Interval': period_key,
                'Start Time': time_info['start'],
                'End Time': time_info['end'],
                'Duration (min)': time_info['duration_minutes']
            }
            
            # Add metrics for each line
            for line_name, line_data in metrics.items():
                if line_name in ['line_distances', 'team']:
                    continue
                    
                for metric_name in ['avg_distance', 'avg_horizontal_distance', 
                                   'avg_vertical_distance', 'x_spread', 'y_spread']:
                    if metric_name in line_data:
                        row[f"{line_name}_{metric_name}"] = line_data[metric_name]
            
            interval_data.append(row)
    
    if interval_data:
        intervals_df = pd.DataFrame(interval_data)
        st.dataframe(intervals_df)
        
        # Add download button
        csv = intervals_df.to_csv(index=False)
        st.download_button(
            label="Download Interval Data CSV",
            data=csv,
            file_name="interval_metrics.csv",
            mime="text/csv"
        )

def display_team_compactness_analysis(period_metrics):
    """Display team compactness metrics over time"""
    st.subheader("Team Compactness Over Time")
    
    # Extract team compactness metrics over time
    periods = []
    width_values = []
    depth_values = []
    area_values = []
    
    for period_key, metrics in period_metrics.items():
        # Skip intervals to keep chart readable
        if period_key.startswith('interval_'):
            continue
            
        if 'team' in metrics:
            team_data = metrics['team']
            periods.append(period_key.replace('_', ' ').title())
            width_values.append(team_data.get('x_spread', 0))
            depth_values.append(team_data.get('y_spread', 0))
            area_values.append(team_data.get('area', 0))
    
    # Create compactness chart
    if periods:
        compact_fig = go.Figure()
        palette = SoccermentPalette()
        
        compact_fig.add_trace(
            go.Bar(
                x=periods,
                y=width_values,
                name='Team Width',
                marker_color=palette.bright_blue
            )
        )
        
        compact_fig.add_trace(
            go.Bar(
                x=periods,
                y=depth_values,
                name='Team Depth',
                marker_color=palette.medium_green
            )
        )
        
        compact_fig.add_trace(
            go.Scatter(
                x=periods,
                y=area_values,
                mode='lines+markers',
                name='Team Area (sq m)',
                line=dict(color=palette.sky_blue, width=3),
                marker=dict(size=10),
                yaxis='y2'
            )
        )
        
        # Update layout with dual y-axis
        compact_fig.update_layout(
            title="Team Compactness Metrics Over Time",
            xaxis_title="Match Period",
            yaxis_title="Distance (m)",
            yaxis2=dict(
                title="Area (sq m)",
                overlaying='y',
                side='right'
            ),
            template="plotly_dark",
            height=500,
            paper_bgcolor=palette.charcoal,
            plot_bgcolor=palette.deep_navy,
            font=dict(color=palette.off_white),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(compact_fig, use_container_width=True)
    else:
        st.info("Insufficient data to analyze team compactness over time.")

def provide_sample_data_downloads():
    """Provide sample data files for download"""
    st.markdown("### Download Sample Data Templates")
    
    sample_tracking = """video_time,x,y,flag_alert,speed,local_time,player_name
0.1,36.43,49.93,0,1.87,2024-12-08 10:15:40.100000,O. Bitte Essoukan
0.1,27.34,54.11,0,13.55,2024-12-08 10:15:40.100000,S. Keita
0.1,39.47,55.75,0,4.65,2024-12-08 10:15:40.100000,L. Garcia
0.1,24.15,43.87,0,3.12,2024-12-08 10:15:40.100000,A. Romagnoli
0.1,70.24,66.51,0,8.32,2024-12-08 10:15:40.100000,D. Calabria"""
    
    st.download_button(
        label="Download Sample Tracking Data CSV",
        data=sample_tracking,
        file_name="sample_tracking_data.csv",
        mime="text/csv"
    )
    
    sample_formation = """team_name,player_id,first_name,last_name,known_name,shirt_number,position,status,module_position
Milan,1,Olivier,Giroud,O. Giroud,9,ST,Active,9.0
Milan,2,Sandro,Tonali,S. Tonali,8,CM,Active,8.0
Milan,3,Rafael,Leao,R. Leao,17,LW,Active,11.0
Milan,4,Davide,Calabria,D. Calabria,2,RB,Active,2.0
Milan,5,Alessio,Romagnoli,A. Romagnoli,13,CB,Active,5.0"""
    
    st.download_button(
        label="Download Sample Formation Data CSV",
        data=sample_formation,
        file_name="sample_formation_data.csv",
        mime="text/csv"
    )

def display_stretch_index_tab(team_manager, selected_period):
    """Display Stretch Index analysis in a tab"""
    st.subheader("Team Stretch Index Analysis")
    
    # Create stretch index analyzer instance
    analyzer = StretchIndexAnalyzer(team_manager.tracking_data, palette=SoccermentPalette())
    
    # Set time periods from team manager
    analyzer.set_time_periods(team_manager.time_periods)
    
    # Compute stretch index with progress indicator
    with st.spinner("Calculating Stretch Index..."):
        stretch_data = analyzer.compute_stretch_index_time_series(exclude_goalkeepers=True)
    
    if stretch_data is not None and not stretch_data.empty:
        # Show stretch index plot for selected period
        st.subheader(f"Stretch Index Time Series - {selected_period.replace('_', ' ').title()}")
        
        # Add options for plotting
        col1, col2 = st.columns(2)
        with col1:
            highlight_peaks = st.checkbox("Highlight Peaks & Dips", value=True)
        
        # Create the plot with the selected options
        si_plot = analyzer.create_stretch_index_plot(
            period_key=selected_period,
            highlight_peaks=highlight_peaks
        )
        
        if si_plot:
            st.plotly_chart(si_plot, use_container_width=True)
        else:
            st.warning("Insufficient data for Stretch Index plot.")
        
        # Display stretch metrics for the period
        metrics = analyzer.get_period_stretch_metrics(selected_period)
        if metrics:
            st.subheader("Stretch Index Metrics")
            
            # Create metrics display in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Stretch Index", f"{metrics['mean_si']:.2f} m")
                st.metric("Min Stretch Index", f"{metrics['min_si']:.2f} m")
            
            with col2:
                st.metric("Median Stretch Index", f"{metrics['median_si']:.2f} m")
                st.metric("Max Stretch Index", f"{metrics['max_si']:.2f} m")
            
            with col3:
                st.metric("Variability (Std Dev)", f"{metrics['std_si']:.2f}")
                if 'sample_entropy' in metrics:
                    st.metric("Sample Entropy", f"{metrics['sample_entropy']:.3f}")
        
        # Show period comparison chart
        st.subheader("Period Comparison")
        period_chart = analyzer.create_period_comparison_chart()
        if period_chart:
            st.plotly_chart(period_chart, use_container_width=True)
        
        # Show centroid movement
        st.subheader("Team Centroid Movement (Animated)")
        centroid_plot = analyzer.create_animated_centroid_plot(selected_period)
        if centroid_plot:
            st.plotly_chart(centroid_plot, use_container_width=True)
            
            # Add explanation of centroid
            with st.expander("About Team Centroid"):
                st.markdown("""
                The team centroid represents the geometric center of all players on the pitch.
                It shows the collective positioning of the team over time.
                
                **Interpretation:**
                - Path shows team movement
                - When combined with Stretch Index, provides insights into:
                  - Team expansion during attacking phases
                  - Team compression during defensive phases
                  - Collective pressing and defensive strategies
                """)
        
        # Add export option for stretch index data
        with st.expander("Export Stretch Index Data"):
            # Allow downloading the stretch index data
            csv = stretch_data.to_csv(index=False)
            st.download_button(
                label="Download Stretch Index CSV",
                data=csv,
                file_name=f"stretch_index_{selected_period}.csv",
                mime="text/csv"
            )
    else:
        st.warning("No data available for Stretch Index analysis. Please check player tracking data.")
    
    # Add explanatory information
    with st.expander("About Stretch Index"):
        st.markdown("""
        ## Stretch Index Explained
        
        **Definition:** The Stretch Index (SI) measures the average distance of players from the team's geometric center (centroid).
        
        **Formula:** SI = (1/N) × Σ di
        - N = Number of players
        - di = Distance of player i from team centroid
        
        **Interpretation:**
        - **Higher values:** Team is more spread out (typically during attacking phases)
        - **Lower values:** Team is more compact (typically during defensive phases)
        
        **Research Context:**
        The Stretch Index has been established in sports science literature (Clemente et al., 2013; Coutinho et al., 2019b; Olthof et al., 2018; Praça et al., 2021) as a useful metric for analyzing team tactical behavior.
        
        **Applications:**
        - Measuring team compactness/expansion during different match phases
        - Analyzing tactical adjustments between periods
        - Comparing different playing styles and formations
        """)

def display_help_information():
    """Display help and sample data information when no data is loaded"""
    st.info("Please load player tracking and team formation data to begin analysis.")
    
    # Display sample data format
    with st.expander("Expected Data Format"):
        st.markdown("### Tracking Data Format")
        st.markdown("""
        The tracking data should be a CSV file with the following columns:
        - `video_time`: Time in seconds
        - `x`: X-coordinate (0-100 scale)
        - `y`: Y-coordinate (0-100 scale)
        - `flag_alert`: Alert flag (0 or 1)
        - `speed`: Player speed in m/s
        - `local_time` or `local_time`: local_time in format "YYYY-MM-DD HH:MM:SS.ssssss"
        - `player_name`: Player name

        Example:
        ```
        video_time,x,y,flag_alert,speed,local_time,player_name
        0.1,36.43,49.93,0,1.87,"2024-12-08 10:15:40.100000","O. Bitte Essoukan"
        0.1,27.34,54.11,0,13.55,"2024-12-08 10:15:40.100000","S. Keita"
        ```
        """)
        
        st.markdown("### Team Formation Data Format")
        st.markdown("""
        The team formation data should be a CSV file with the following columns:
        - `team_name`: Team name
        - `player_id`: Player ID
        - `first_name`: Player's first name
        - `last_name`: Player's last name
        - `known_name`: Player's common name (how the player is commonly referred to)
        - `shirt_number`: Jersey number
        - `position`: Player position code (e.g., GK, CB, CM)
        - `status`: Player status (Active, Injured, etc.)
        - `module_position`: Position in formation module (numerical position in tactical setup)

        Example:
        ```
        team_name,player_id,first_name,last_name,known_name,shirt_number,position,status,module_position
        Milan,1,Olivier,Giroud,O. Giroud,9,ST,Active,9.0
        Milan,2,Sandro,Tonali,S. Tonali,8,CM,Active,8.0
        ```
        """)
        
        # Common troubleshooting tips
        st.markdown("### Troubleshooting")
        st.markdown("""
        If you encounter issues loading your data:
        
        1. **local_time Format**: Ensure your local_time column is in a standard format (YYYY-MM-DD HH:MM:SS)
        2. **Column Names**: Check that column names match the expected format (case-sensitive)
        3. **CSV Format**: Make sure your CSV is properly formatted with consistent delimiters
        4. **Special Characters**: Avoid special characters in player names or use quotes
        5. **Timezone Issues**: Be aware of potential timezone differences in local_time
        6. **Missing Values**: Fill in missing values or ensure your data handling can manage them
        """)
        
        # Add sample CSV download
        st.markdown("### Download Sample Data Templates")
        
        sample_tracking = """video_time,x,y,flag_alert,speed,local_time,player_name
0.1,36.43,49.93,0,1.87,2024-12-08 10:15:40.100000,O. Bitte Essoukan
0.1,27.34,54.11,0,13.55,2024-12-08 10:15:40.100000,S. Keita
0.1,39.47,55.75,0,4.65,2024-12-08 10:15:40.100000,L. Garcia
0.1,24.15,43.87,0,3.12,2024-12-08 10:15:40.100000,A. Romagnoli
0.1,70.24,66.51,0,8.32,2024-12-08 10:15:40.100000,D. Calabria
0.2,36.58,50.21,0,2.01,2024-12-08 10:15:40.200000,O. Bitte Essoukan
0.2,27.58,54.38,0,13.41,2024-12-08 10:15:40.200000,S. Keita
0.2,39.73,55.93,0,4.54,2024-12-08 10:15:40.200000,L. Garcia
0.2,24.23,44.02,0,3.23,2024-12-08 10:15:40.200000,A. Romagnoli
0.2,70.38,66.68,0,8.12,2024-12-08 10:15:40.200000,D. Calabria"""
        
        st.download_button(
            label="Download Sample Tracking Data CSV",
            data=sample_tracking,
            file_name="sample_tracking_data.csv",
            mime="text/csv"
        )
        
        sample_formation = """team_name,player_id,first_name,last_name,known_name,shirt_number,position,status,module_position
Milan,1,Olivier,Giroud,O. Giroud,9,ST,Active,9.0
Milan,2,Sandro,Tonali,S. Tonali,8,CM,Active,8.0
Milan,3,Rafael,Leao,R. Leao,17,LW,Active,11.0
Milan,4,Davide,Calabria,D. Calabria,2,RB,Active,2.0
Milan,5,Alessio,Romagnoli,A. Romagnoli,13,CB,Active,5.0
Milan,6,Luis,Garcia,L. Garcia,14,CM,Active,6.0
Milan,7,Mike,Maignan,M. Maignan,16,GK,Active,1.0
Milan,8,Simon,Kjaer,S. Kjaer,24,CB,Active,4.0
Milan,9,Fikayo,Tomori,F. Tomori,23,CB,Active,5.0
Milan,10,Theo,Hernandez,T. Hernandez,19,LB,Active,3.0
Milan,11,Alexis,Saelemaekers,A. Saelemaekers,56,RW,Active,7.0"""
        
        st.download_button(
            label="Download Sample Formation Data CSV",
            data=sample_formation,
            file_name="sample_formation_data.csv",
            mime="text/csv"
        )
        
        # Add information about the app's features
        st.markdown("### App Features")
        st.markdown("""
        This app allows you to:
        
        1. **Team Formation Visualization**: See player positions on the pitch with tactical lines
        2. **Distance Metrics**: Analyze distances between players and team lines
        3. **Positional Analysis**: Examine individual player movements and create heatmaps
        4. **Time Series Analysis**: Track how team formation changes throughout the match
        5. **Stretch Index Analysis**: Measure team expansion and compactness based on scientific literature
        
        To get started, upload your tracking and formation data files or provide file paths, then click "Load Data".
        """)        
# Run the main function when the script is executed
if __name__ == "__main__":
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    main()

