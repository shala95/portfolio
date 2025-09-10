import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
import nolds  # For entropy calculation (optional)

class StretchIndexAnalyzer:
    """
    A class to calculate and analyze the Stretch Index of a football team
    based on positional tracking data.
    
    Based on methodology from:
    - Clemente et al., 2013
    - Coutinho et al., 2019b
    - Olthof et al., 2018
    - Praça et al., 2021
    """
    
    def __init__(self, tracking_data: pd.DataFrame, palette=None):
        """
        Initialize the Stretch Index Analyzer.
        
        Parameters:
        -----------
        tracking_data : pd.DataFrame
            Tracking data with columns: 'local_time', 'player_name', 'x', 'y'
        palette : object, optional
            Color palette object for visualization consistency
        """
        self.tracking_data = tracking_data
        self.palette = palette
        self.stretch_index_results = None
        self.time_periods = {}
    
    def set_time_periods(self, time_periods: Dict):
        """Set the time periods to use for analysis segmentation."""
        self.time_periods = time_periods
    
    def exclude_players(self, players_to_exclude: List[str] = None) -> pd.DataFrame:
        """
        Filter out players that should be excluded from the Stretch Index calculation.
        By default, removes goalkeepers.
        
        Parameters:
        -----------
        players_to_exclude : List[str], optional
            List of player names to exclude from calculation
            
        Returns:
        --------
        pd.DataFrame
            Filtered tracking data
        """
        if players_to_exclude is None:
            players_to_exclude = []
            
            # Try to automatically identify goalkeepers based on position text in column name
            if 'position' in self.tracking_data.columns:
                goalkeeper_mask = self.tracking_data['position'].str.contains('GK', case=False, na=False)
                goalkeepers = self.tracking_data.loc[goalkeeper_mask, 'player_name'].unique().tolist()
                players_to_exclude.extend(goalkeepers)
        
        if players_to_exclude:
            return self.tracking_data[~self.tracking_data['player_name'].isin(players_to_exclude)]
        else:
            return self.tracking_data
    
    def calculate_team_centroid(self, frame_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate the team centroid (geometric center) for a given frame.
        
        Parameters:
        -----------
        frame_data : pd.DataFrame
            Data for a single timestamp/frame with player positions
            
        Returns:
        --------
        Tuple[float, float]
            (centroid_x, centroid_y) coordinates
        """
        if frame_data.empty:
            return (np.nan, np.nan)
        
        # Calculate mean x and y positions
        centroid_x = frame_data['x'].mean()
        centroid_y = frame_data['y'].mean()
        
        return (centroid_x, centroid_y)
    
    def calculate_distances_to_centroid(self, frame_data: pd.DataFrame, 
                                       centroid: Tuple[float, float]) -> List[float]:
        """
        Calculate the Euclidean distance from each player to the team centroid.
        
        Parameters:
        -----------
        frame_data : pd.DataFrame
            Data for a single timestamp/frame with player positions
        centroid : Tuple[float, float]
            (centroid_x, centroid_y) coordinates
            
        Returns:
        --------
        List[float]
            List of distances from each player to the centroid
        """
        if frame_data.empty or np.isnan(centroid[0]) or np.isnan(centroid[1]):
            return []
        
        # Calculate Euclidean distance for each player to centroid
        distances = []
        for _, player in frame_data.iterrows():
            dist = math.sqrt((player['x'] - centroid[0])**2 + (player['y'] - centroid[1])**2)
            distances.append(dist)
        
        return distances
    
    def calculate_stretch_index(self, distances: List[float]) -> float:
        """
        Calculate the Stretch Index for a frame.
        
        Parameters:
        -----------
        distances : List[float]
            List of distances from each player to the centroid
            
        Returns:
        --------
        float
            Stretch Index value (mean distance to centroid)
        """
        if not distances:
            return np.nan
        
        # Calculate mean distance to centroid (Stretch Index)
        stretch_index = sum(distances) / len(distances)
        return stretch_index
    
    def compute_stretch_index_time_series(self, exclude_goalkeepers: bool = True, 
                                         min_players: int = 7) -> pd.DataFrame:
        """
        Compute the Stretch Index time series for the full match.
        
        Parameters:
        -----------
        exclude_goalkeepers : bool, optional
            Whether to exclude goalkeepers from calculations
        min_players : int, optional
            Minimum number of players required for a valid calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns 'local_time', 'stretch_index'
        """
        # Filter out goalkeepers if required
        filtered_data = self.exclude_players() if exclude_goalkeepers else self.tracking_data
        
        # Group by timestamp
        grouped_by_time = filtered_data.groupby('local_time')
        
        # Calculate Stretch Index for each timestamp
        results = []
        for time, frame_data in grouped_by_time:
            # Skip if not enough players
            if len(frame_data) < min_players:
                continue
                
            # Calculate centroid
            centroid = self.calculate_team_centroid(frame_data)
            
            # Calculate distances to centroid
            distances = self.calculate_distances_to_centroid(frame_data, centroid)
            
            # Calculate Stretch Index
            si = self.calculate_stretch_index(distances)
            
            # Store results
            results.append({
                'local_time': time,
                'stretch_index': si,
                'player_count': len(frame_data),
                'centroid_x': centroid[0],
                'centroid_y': centroid[1]
            })
        
        # Create DataFrame from results
        self.stretch_index_results = pd.DataFrame(results)
        return self.stretch_index_results
    
    def get_period_stretch_metrics(self, period_key: str = 'full_match') -> Dict:
        """
        Calculate stretch index statistics for a specific time period.
        
        Parameters:
        -----------
        period_key : str, optional
            Key of the time period to analyze
            
        Returns:
        --------
        Dict
            Dictionary of stretch index metrics for the period
        """
        if self.stretch_index_results is None:
            return {}
            
        if period_key not in self.time_periods:
            return {}
            
        # Get period time range
        period = self.time_periods[period_key]
        start_time = period['start']
        end_time = period['end']
        
        # Filter data for the period
        period_data = self.stretch_index_results[
            (self.stretch_index_results['local_time'] >= start_time) & 
            (self.stretch_index_results['local_time'] <= end_time)
        ]
        
        if period_data.empty:
            return {}
            
        # Calculate metrics
        metrics = {
            'mean_si': period_data['stretch_index'].mean(),
            'median_si': period_data['stretch_index'].median(),
            'max_si': period_data['stretch_index'].max(),
            'min_si': period_data['stretch_index'].min(),
            'std_si': period_data['stretch_index'].std(),
            'data_points': len(period_data)
        }
        
        # Add timestamps for max and min SI
        max_si_idx = period_data['stretch_index'].idxmax()
        min_si_idx = period_data['stretch_index'].idxmin()
        
        if not pd.isna(max_si_idx):
            metrics['max_si_time'] = period_data.loc[max_si_idx, 'local_time']
        
        if not pd.isna(min_si_idx):
            metrics['min_si_time'] = period_data.loc[min_si_idx, 'local_time']
        
        # Try to calculate entropy if enough data points
        try:
            if len(period_data) > 100:  # Need sufficient data for entropy calculation
                # Sample Entropy calculation (from nolds package)
                sample_entropy = nolds.sampen(period_data['stretch_index'].dropna().values, emb_dim=2)
                metrics['sample_entropy'] = sample_entropy
        except:
            # Entropy calculation is optional
            pass
            
        return metrics
    
    def calculate_all_period_metrics(self) -> Dict:
        """
        Calculate stretch index metrics for all defined time periods.
        
        Returns:
        --------
        Dict
            Dictionary with period keys and their metrics
        """
        all_metrics = {}
        
        for period_key in self.time_periods.keys():
            metrics = self.get_period_stretch_metrics(period_key)
            if metrics:
                all_metrics[period_key] = metrics
                
        return all_metrics
    
    def create_stretch_index_plot(self, period_key='full_match', 
                              highlight_peaks=True) -> go.Figure:
        """
        Create an enhanced line plot visualization of the Stretch Index over time,
        ensuring complete period coverage.
        
        Parameters:
        -----------
        period_key : str, optional
            Key of the time period to visualize
        highlight_peaks : bool, optional
            Whether to highlight significant peaks and dips
            
        Returns:
        --------
        go.Figure
            Plotly figure object with the stretch index plot
        """
        if self.stretch_index_results is None or self.stretch_index_results.empty:
            return None
            
        if period_key not in self.time_periods:
            return None
            
        # Get period time range
        period = self.time_periods[period_key]
        start_time = period['start']
        end_time = period['end']
        
        # Filter data for the period with a small buffer to ensure complete coverage
        # This helps capture the exact start and end points
        buffer_seconds = 1  # 1 second buffer
        buffer_start = start_time - pd.Timedelta(seconds=buffer_seconds)
        buffer_end = end_time + pd.Timedelta(seconds=buffer_seconds)
        
        # First get all data within the buffer
        buffer_data = self.stretch_index_results[
            (self.stretch_index_results['local_time'] >= buffer_start) & 
            (self.stretch_index_results['local_time'] <= buffer_end)
        ]
        
        # Then filter to exact period and mark the start/end points
        period_data = buffer_data[
            (buffer_data['local_time'] >= start_time) & 
            (buffer_data['local_time'] <= end_time)
        ]
        
        if period_data.empty:
            return None
        
        # Add period start and end percentage markers for context
        period_data['percent_complete'] = ((period_data['local_time'] - start_time) / 
                                        (end_time - start_time) * 100)
        
        # Create line plot
        fig = go.Figure()
        
        # Use palette if available, otherwise default colors
        line_color = self.palette.bright_blue if self.palette else '#1C82BE'
        bg_color = self.palette.charcoal if self.palette else '#181F2A'
        plot_bg = self.palette.deep_navy if self.palette else '#1B2C3B'
        font_color = self.palette.off_white if self.palette else '#EDEDED'
        
        # Add main SI line
        fig.add_trace(
            go.Scatter(
                x=period_data['local_time'],
                y=period_data['stretch_index'],
                mode='lines',
                name='Stretch Index',
                line=dict(width=2, color=line_color),
                hovertemplate='Time: %{x}<br>Stretch Index: %{y:.2f}m<br>Progress: %{customdata:.1f}%',
                customdata=period_data['percent_complete']
            )
        )
        
        # Add mean SI reference line
        mean_si = period_data['stretch_index'].mean()
        fig.add_trace(
            go.Scatter(
                x=[period_data['local_time'].min(), period_data['local_time'].max()],
                y=[mean_si, mean_si],
                mode='lines',
                name=f'Mean SI: {mean_si:.2f}m',
                line=dict(width=1, color='rgba(255,255,255,0.5)', dash='dash')
            )
        )
        
        # Add start and end markers for clear period boundaries
        fig.add_trace(
            go.Scatter(
                x=[start_time, end_time],
                y=[period_data['stretch_index'].iloc[0], period_data['stretch_index'].iloc[-1]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='rgba(255,255,255,0.8)',
                    symbol='circle',
                    line=dict(width=2, color='rgba(0,0,0,0.8)')
                ),
                name='Period Boundaries',
                hovertemplate='%{text}',
                text=['Start (0%)', 'End (100%)']
            )
        )
        
        # Highlight peaks and dips if requested
        if highlight_peaks:
            # Find peaks (high SI values - team stretched)
            # Using simple threshold approach for demonstration
            threshold_high = mean_si + (1.5 * period_data['stretch_index'].std())
            peaks = period_data[period_data['stretch_index'] >= threshold_high]
            
            # Find dips (low SI values - team compact)
            threshold_low = mean_si - (1.0 * period_data['stretch_index'].std())
            dips = period_data[period_data['stretch_index'] <= threshold_low]
            
            # Add peak markers with percentage information
            if not peaks.empty:
                fig.add_trace(
                    go.Scatter(
                        x=peaks['local_time'],
                        y=peaks['stretch_index'],
                        mode='markers',
                        name='Team Stretched',
                        marker=dict(
                            size=10,
                            color='#ff6b6b',  # Red for peaks
                            symbol='triangle-up'
                        ),
                        hovertemplate='Time: %{x}<br>Stretch Index: %{y:.2f}m<br>Progress: %{customdata:.1f}%',
                        customdata=peaks['percent_complete']
                    )
                )
            
            # Add dip markers with percentage information
            if not dips.empty:
                fig.add_trace(
                    go.Scatter(
                        x=dips['local_time'],
                        y=dips['stretch_index'],
                        mode='markers',
                        name='Team Compact',
                        marker=dict(
                            size=10,
                            color='#4ecdc4',  # Teal for dips
                            symbol='triangle-down'
                        ),
                        hovertemplate='Time: %{x}<br>Stretch Index: %{y:.2f}m<br>Progress: %{customdata:.1f}%',
                        customdata=dips['percent_complete']
                    )
                )
        
        # Add time percentage reference lines
        if period_data['percent_complete'].max() > 25:  # Only add if period is long enough
            percentiles = [25, 50, 75]
            for percentile in percentiles:
                # Find closest data point to the percentile
                target_time = start_time + pd.Timedelta(seconds=(end_time - start_time).total_seconds() * percentile / 100)
                closest_idx = (period_data['local_time'] - target_time).abs().idxmin()
                
                fig.add_trace(
                    go.Scatter(
                        x=[period_data.loc[closest_idx, 'local_time'], period_data.loc[closest_idx, 'local_time']],
                        y=[period_data['stretch_index'].min() * 0.95, period_data['stretch_index'].max() * 1.05],
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
                        name=f'{percentile}% of Period',
                        hoverinfo='name'
                    )
                )
        
        # Update layout
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        fig.update_layout(
            title={
                'text': f"Team Stretch Index - {period_key.replace('_', ' ').title()} ({duration_minutes:.1f} min)",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': "Time (0% → 100% of Period)",
                'standoff': 15
            },
            yaxis_title="Stretch Index (m)",
            template="plotly_dark",
            height=500,
            paper_bgcolor=bg_color,
            plot_bgcolor=plot_bg,
            font=dict(color=font_color),
            # Add annotations to explain the graph
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=f"Full Period Coverage: {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')} (Duration: {duration_minutes:.1f} min)",
                    showarrow=False,
                    font=dict(color=font_color, size=12)
                )
            ],
            # Improve legend positioning
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add hover mode for better user interaction
            hovermode="closest"
        )
        
        return fig
        
    def create_period_comparison_chart(self) -> go.Figure:
        """
        Create a bar chart comparing stretch index across different periods.
        
        Returns:
        --------
        go.Figure
            Plotly figure with period comparison
        """
        period_metrics = self.calculate_all_period_metrics()
        
        if not period_metrics:
            return None
            
        # Prepare data for chart
        periods = []
        mean_values = []
        std_values = []
        
        for period_key, metrics in period_metrics.items():
            # Skip intervals for clarity in the chart
            if period_key.startswith('interval_'):
                continue
                
            periods.append(period_key.replace('_', ' ').title())
            mean_values.append(metrics['mean_si'])
            std_values.append(metrics['std_si'])
        
        # Create figure
        fig = go.Figure()
        
        # Use palette if available
        bar_color = self.palette.bright_blue if self.palette else '#1C82BE'
        error_color = self.palette.steel_blue if self.palette else '#1D83BF'
        bg_color = self.palette.charcoal if self.palette else '#181F2A'
        plot_bg = self.palette.deep_navy if self.palette else '#1B2C3B'
        font_color = self.palette.off_white if self.palette else '#EDEDED'
        
        # Add bar chart with error bars
        fig.add_trace(
            go.Bar(
                x=periods,
                y=mean_values,
                error_y=dict(
                    type='data',
                    array=std_values,
                    visible=True,
                    color=error_color
                ),
                marker_color=bar_color,
                text=[f"{v:.2f}" for v in mean_values],
                textposition="auto"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Mean Stretch Index by Period",
            xaxis_title="Match Period",
            yaxis_title="Mean Stretch Index (m)",
            template="plotly_dark",
            height=500,
            paper_bgcolor=bg_color,
            plot_bgcolor=plot_bg,
            font=dict(color=font_color)
        )
        
        return fig
    
    def create_animated_centroid_plot(self, period_key='full_match', playback_speed=1.0, include_players=True):
        """
        Create an enhanced animated plot showing team centroid movement with stretch index values
        and individual player positions.
        
        Parameters:
        -----------
        period_key : str, optional
            Key of the time period to visualize
        playback_speed : float, optional
            Speed multiplier for animation (1.0 = real-time, 2.0 = 2x speed)
        include_players : bool, optional
            Whether to include individual player positions in the visualization
            
        Returns:
        --------
        go.Figure
            Plotly figure with animated centroid movement and stretch index visualization
        """
        if self.stretch_index_results is None or self.stretch_index_results.empty:
            return None
            
        if period_key not in self.time_periods:
            return None
            
        # Get period time range
        period = self.time_periods[period_key]
        start_time = period['start']
        end_time = period['end']
        
        # Calculate actual period duration in seconds
        period_duration_seconds = (end_time - start_time).total_seconds()
        
        # Filter centroid data for the period
        period_data = self.stretch_index_results[
            (self.stretch_index_results['local_time'] >= start_time) & 
            (self.stretch_index_results['local_time'] <= end_time)
        ].copy()
        
        if period_data.empty:
            return None
        
        # Also filter the tracking data if we're including players
        # Assuming self.tracking_data contains the full tracking dataset
        if include_players and hasattr(self, 'tracking_data') and self.tracking_data is not None:
            player_data = self.tracking_data[
                (self.tracking_data['local_time'] >= start_time) & 
                (self.tracking_data['local_time'] <= end_time)
            ].copy()
            
            # Get unique players
            players = player_data['player_name'].unique() if 'player_name' in player_data.columns else []
            
            # Get position categories for color-coding
            player_categories = {}
            for player in players:
                # Try to determine player category from position
                if 'position' in player_data.columns:
                    positions = player_data[player_data['player_name'] == player]['position'].unique()
                    position = positions[0] if len(positions) > 0 else ""
                    
                    # Assign category based on position
                    if any(gk in position.lower() for gk in ['gk', 'goal', 'keeper']):
                        category = 'Goalkeeper'
                    elif any(def_pos in position.lower() for def_pos in ['cb', 'lb', 'rb', 'def']):
                        category = 'Defender'
                    elif any(mid_pos in position.lower() for mid_pos in ['cm', 'cdm', 'cam', 'mid']):
                        category = 'Midfielder'
                    elif any(att_pos in position.lower() for att_pos in ['st', 'cf', 'lw', 'rw', 'att']):
                        category = 'Forward'
                    else:
                        category = 'Unknown'
                    
                    player_categories[player] = category
                else:
                    player_categories[player] = 'Unknown'
        else:
            player_data = None
            players = []
            player_categories = {}
        
        # Create the pitch layout
        bg_color = self.palette.charcoal if self.palette else '#181F2A'
        pitch_color = self.palette.deep_navy if self.palette else '#1B2C3B'
        line_color = self.palette.off_white if self.palette else '#EDEDED'
        
        fig = go.Figure()
        
        # Add the pitch rectangle (main field)
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=100, y1=100,
            line=dict(color=line_color),
            fillcolor=pitch_color,
            layer="below"
        )
        
        # Add center line and other pitch elements
        fig.add_shape(
            type="line",
            x0=50, y0=0, x1=50, y1=100,
            line=dict(color=line_color)
        )
        
        fig.add_shape(
            type="circle",
            x0=40, y0=40, x1=60, y1=60,
            line=dict(color=line_color)
        )
        
        fig.add_shape(
            type="rect",
            x0=0, y0=21.1, x1=16.5, y1=78.9,
            line=dict(color=line_color)
        )
        
        fig.add_shape(
            type="rect",
            x0=83.5, y0=21.1, x1=100, y1=78.9,
            line=dict(color=line_color)
        )
        
        # Define position colors
        position_colors = {
            'Goalkeeper': self.palette.goalkeeper_color if self.palette else '#1C82BE',
            'Defender': self.palette.defender_color if self.palette else '#1D83BF',
            'Midfielder': self.palette.midfielder_color if self.palette else '#2DE598',
            'Forward': self.palette.forward_color if self.palette else '#2CB7D9',
            'Unknown': self.palette.light_grey if self.palette else '#BEBEBE'
        }
        
        # Create evenly distributed frames across the entire period
        if period_duration_seconds <= 300:  # 5 minutes or less
            num_frames = 100
        else:
            num_frames = min(200, max(100, int(period_duration_seconds / 3)))
        
        # Create timestamps at regular intervals
        timestamps = [
            start_time + pd.Timedelta(seconds=i * period_duration_seconds / (num_frames - 1))
            for i in range(num_frames)
        ]
        
        # Ensure the last timestamp is exactly the end_time
        timestamps[-1] = end_time
        
        # For each timestamp, find the nearest centroid data point
        animation_data = []
        for ts in timestamps:
            # Find closest centroid data point
            closest_idx = (period_data['local_time'] - ts).abs().idxmin()
            closest_point = period_data.loc[closest_idx].copy()
            
            # Create a row with the exact timestamp but data from the closest point
            row = closest_point.copy()
            row['original_time'] = row['local_time']  # Keep original for reference
            row['local_time'] = ts  # Use the evenly spaced timestamp
            
            # Add elapsed time and percentage for better labeling
            row['elapsed_seconds'] = (ts - start_time).total_seconds()
            row['percent_complete'] = (row['elapsed_seconds'] / period_duration_seconds) * 100
            
            animation_data.append(row)
        
        # Convert to DataFrame
        animation_df = pd.DataFrame(animation_data)
        
        # Get min/max stretch index for consistent colorscale
        min_si = animation_df['stretch_index'].min()
        max_si = animation_df['stretch_index'].max()
        
        # Create a trace for the centroid point that will be animated
        fig.add_trace(
            go.Scatter(
                x=[animation_df.iloc[0]['centroid_x']],
                y=[animation_df.iloc[0]['centroid_y']],
                mode='markers',
                marker=dict(
                    size=15,
                    color=[animation_df.iloc[0]['stretch_index']],
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Stretch Index (m)",
                        titleside="right"
                    ),
                    cmin=min_si,
                    cmax=max_si,
                    showscale=True
                ),
                name="Team Centroid"
            )
        )
        
        # Create a trace for the path that will be built up over time
        fig.add_trace(
            go.Scatter(
                x=[animation_df.iloc[0]['centroid_x']],
                y=[animation_df.iloc[0]['centroid_y']],
                mode='lines',
                line=dict(
                    width=2,
                    color=self.palette.medium_blue if self.palette else '#1BB3D7'
                ),
                name="Centroid Path"
            )
        )
        
        # Add a text annotation for the current stretch index value
        fig.add_trace(
            go.Scatter(
                x=[5],  # Fixed position in the corner
                y=[5],
                mode='text',
                text=[f"Stretch Index: {animation_df.iloc[0]['stretch_index']:.2f} m"],
                textfont=dict(
                    size=16,
                    color="white"
                ),
                name="SI Value",
                showlegend=False
            )
        )
        
        # Add a text for the confidence indication
        fig.add_trace(
            go.Scatter(
                x=[5],  # Fixed position
                y=[95],  # Top left corner
                mode='text',
                text=["Confidence: High"],  # Default to high, will update in frames
                textfont=dict(
                    size=14,
                    color="white"
                ),
                name="Confidence",
                showlegend=False
            )
        )
        
        # Add player positions for the first frame if include_players is True
        player_traces = []
        if include_players and player_data is not None and not player_data.empty:
            # Get player positions for the first timestamp
            first_frame_time = timestamps[0]
            time_window = 0.5  # seconds on either side to find players
            
            # Find player data points closest to the first timestamp
            first_frame_players = player_data[
                (player_data['local_time'] >= first_frame_time - pd.Timedelta(seconds=time_window)) &
                (player_data['local_time'] <= first_frame_time + pd.Timedelta(seconds=time_window))
            ]
            
            # Group by player and get latest position for each
            latest_positions = {}
            for player in players:
                player_points = first_frame_players[first_frame_players['player_name'] == player]
                if not player_points.empty:
                    # Get point closest to the timestamp
                    closest_idx = (player_points['local_time'] - first_frame_time).abs().idxmin()
                    latest_positions[player] = player_points.loc[closest_idx]
            
            # Add a trace for each player
            for player, pos in latest_positions.items():
                # Get category and color
                category = player_categories.get(player, 'Unknown')
                color = position_colors.get(category, position_colors['Unknown'])
                
                # Add player marker
                player_trace = go.Scatter(
                    x=[pos['x']],
                    y=[pos['y']],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    text=[player.split()[-1] if len(player.split()) > 1 else player],  # Show last name or initials
                    textposition="top center",
                    textfont=dict(
                        family="Arial",
                        size=10,
                        color="white"
                    ),
                    name=f"{player} ({category})",
                    hoverinfo="name+text",
                    hovertext=f"{player} - {category}"
                )
                fig.add_trace(player_trace)
                player_traces.append(player_trace)
        
        # Build animation frames
        frames = []
        for i in range(1, len(animation_df)):
            # Get all centroid positions up to this point
            path_x = animation_df.iloc[:i+1]['centroid_x'].tolist()
            path_y = animation_df.iloc[:i+1]['centroid_y'].tolist()
            
            # Current position and stretch index
            current_x = animation_df.iloc[i]['centroid_x']
            current_y = animation_df.iloc[i]['centroid_y']
            current_si = animation_df.iloc[i]['stretch_index']
            current_time = animation_df.iloc[i]['local_time']
            
            # Determine confidence based on stretch index and centroid position
            confidence_text = "Confidence: "
            if (current_x < 0 or current_x > 100 or current_y < 0 or current_y > 100 or
                current_si > max_si * 0.9 or current_si < min_si * 1.1):
                confidence_text += "Low"
                confidence_color = "red"
            elif (current_si > np.percentile(animation_df['stretch_index'], 75) or 
                current_si < np.percentile(animation_df['stretch_index'], 25)):
                confidence_text += "Medium"
                confidence_color = "yellow"
            else:
                confidence_text += "High"
                confidence_color = "green"
                
            # Player positions for this frame
            frame_player_data = []
            if include_players and player_data is not None:
                # Get timestamp for this frame
                frame_time = timestamps[i]
                
                # Find player positions for this frame
                frame_players = player_data[
                    (player_data['local_time'] >= frame_time - pd.Timedelta(seconds=time_window)) &
                    (player_data['local_time'] <= frame_time + pd.Timedelta(seconds=time_window))
                ]
                
                # Group by player and get latest position for each
                frame_positions = {}
                for player in players:
                    player_points = frame_players[frame_players['player_name'] == player]
                    if not player_points.empty:
                        closest_idx = (player_points['local_time'] - frame_time).abs().idxmin()
                        frame_positions[player] = player_points.loc[closest_idx]
                
                # Create trace data for each player
                for player, pos in frame_positions.items():
                    category = player_categories.get(player, 'Unknown')
                    color = position_colors.get(category, position_colors['Unknown'])
                    
                    frame_player_data.append(
                        go.Scatter(
                            x=[pos['x']],
                            y=[pos['y']],
                            mode='markers+text',
                            marker=dict(
                                size=10,
                                color=color,
                                line=dict(width=1, color='white')
                            ),
                            text=[player.split()[-1] if len(player.split()) > 1 else player],
                            textposition="top center",
                            textfont=dict(
                                family="Arial",
                                size=10,
                                color="white"
                            ),
                            name=f"{player} ({category})",
                            hoverinfo="name+text",
                            hovertext=f"{player} - {category}"
                        )
                    )
            
            # Create the frame with all traces
            frame_data = [
                # Centroid marker
                go.Scatter(
                    x=[current_x],
                    y=[current_y],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=[current_si],
                        colorscale='Viridis',
                        cmin=min_si,
                        cmax=max_si,
                        showscale=True
                    ),
                    name="Team Centroid"
                ),
                # Centroid path
                go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode='lines',
                    line=dict(
                        width=2,
                        color=self.palette.medium_blue if self.palette else '#1BB3D7'
                    ),
                    name="Centroid Path"
                ),
                # Stretch index value text
                go.Scatter(
                    x=[5],
                    y=[5],
                    mode='text',
                    text=[f"Stretch Index: {current_si:.2f} m"],
                    textfont=dict(
                        size=16,
                        color="white"
                    ),
                    name="SI Value",
                    showlegend=False
                ),
                # Confidence indicator
                go.Scatter(
                    x=[5],
                    y=[95],
                    mode='text',
                    text=[confidence_text],
                    textfont=dict(
                        size=14,
                        color=confidence_color
                    ),
                    name="Confidence",
                    showlegend=False
                )
            ]
            
            # Add player traces if available
            if frame_player_data:
                frame_data.extend(frame_player_data)
            
            # Create the frame
            frame = go.Frame(
                data=frame_data,
                name=str(i)
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Calculate animation parameters
        animation_duration_ms = int(period_duration_seconds * 1000 / playback_speed)
        frame_duration = animation_duration_ms / len(frames)
        
        # Create labels with time, percentage and stretch index
        elapsed_times = animation_df['elapsed_seconds'].tolist()
        percent_complete = animation_df['percent_complete'].tolist()
        stretch_index_values = animation_df['stretch_index'].tolist()
        
        time_labels = {
            str(i): f"{elapsed:.1f}s ({pct:.0f}%) - SI: {si:.2f}m" 
            for i, (elapsed, pct, si) in enumerate(zip(elapsed_times, percent_complete, stretch_index_values))
        }
        
        # Add animation controls
        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": frame_duration, "redraw": True}, 
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False}, 
                            "mode": "immediate"
                        }]
                    ),
                    dict(
                        label="Reset",
                        method="animate",
                        args=[["0"], {
                            "frame": {"duration": 0, "redraw": True}, 
                            "mode": "immediate"
                        }]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ]
        
        # Create slider with evenly distributed steps
        num_slider_steps = min(10, len(animation_df))
        slider_indices = [int(i * (len(animation_df) - 1) / (num_slider_steps - 1)) for i in range(num_slider_steps)]
        
        sliders = [
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Time: ",
                    visible=True,
                    xanchor="right"
                ),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True}, 
                                "mode": "immediate"
                            }
                        ],
                        label=time_labels.get(str(i), f"{i}")
                    )
                    for i in slider_indices
                ]
            )
        ]
        
        # Get stats for title and annotations
        total_duration_minutes = period_duration_seconds / 60
        avg_si = animation_df['stretch_index'].mean()
        min_si = animation_df['stretch_index'].min()
        max_si = animation_df['stretch_index'].max()
        
        # Update layout
        fig.update_layout(
            title=f"Team Centroid Movement & Player Positions - {period_key.replace('_', ' ').title()} (Avg SI: {avg_si:.2f}m)",
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
            updatemenus=updatemenus,
            sliders=sliders,
            width=800,
            height=600,
            paper_bgcolor=bg_color,
            plot_bgcolor=pitch_color,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=True,
            legend=dict(
                font=dict(color=line_color),
                bgcolor=bg_color,
                bordercolor=line_color,
                x=1.1,  # Move legend outside the plot area
                y=0.5,
                xanchor="left",
                yanchor="middle"
            ),
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=f"Playback Speed: {playback_speed}x | SI Range: {min_si:.2f}m - {max_si:.2f}m | Players: {len(players)}",
                    showarrow=False,
                    font=dict(color=line_color)
                ),
                dict(
                    x=0.5,
                    y=1.02,
                    xref="paper",
                    yref="paper",
                    text=f"Period: {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')} (Duration: {total_duration_minutes:.1f} min)",
                    showarrow=False,
                    font=dict(color=line_color)
                )
            ]
        )
        
        return fig
def display_stretch_index_tab(team_manager, selected_period):
    """Display enhanced Stretch Index analysis in a tab"""
    st.subheader("Team Stretch Index Analysis")
    
    # Create stretch index analyzer instance
    analyzer = StretchIndexAnalyzer(team_manager.tracking_data, palette=SoccermentPalette())
    
    # Set time periods from team manager
    analyzer.set_time_periods(team_manager.time_periods)
    
    # Compute stretch index with progress indicator
    with st.spinner("Calculating Stretch Index..."):
        stretch_data = analyzer.compute_stretch_index_time_series(exclude_goalkeepers=True)
    
    if stretch_data is not None and not stretch_data.empty:
        # Show period selection context
        period_info = team_manager.time_periods.get(selected_period, {})
        if period_info:
            st.info(
                f"⏱️ Selected Period: {selected_period.replace('_', ' ').title()}\n"
                f"Start: {period_info['start']}\n"
                f"End: {period_info['end']}\n"
                f"Duration: {period_info['duration_minutes']:.2f} minutes"
            )
        
        # Add options for plotting in a more organized layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            highlight_peaks = st.checkbox("Highlight Peaks & Dips", value=True)
        
        with col2:
            show_percentiles = st.checkbox("Show Period Percentiles", value=True)
        
        with col3:
            smoothing = st.checkbox("Apply Smoothing", value=False)
        
        # Show stretch index plot for selected period
        st.subheader(f"Stretch Index Time Series - {selected_period.replace('_', ' ').title()}")
        
        # Create the plot with the selected options
        si_plot = analyzer.create_stretch_index_plot(
            period_key=selected_period,
            highlight_peaks=highlight_peaks
        )
        
        if si_plot:
            st.plotly_chart(si_plot, use_container_width=True)
            
            # Add an explanation of the stretch index plot
            with st.expander("Understanding the Stretch Index Plot"):
                st.markdown("""
                **Reading the Stretch Index Plot:**
                
                - **Main Line**: Shows how the team's spread (average distance from centroid) changes over time
                - **Red Triangles**: Indicate moments when the team is highly stretched out (attacking phases)
                - **Teal Triangles**: Indicate moments when the team is compact (defensive phases)
                - **Dashed Line**: Shows the average stretch index value for this period
                - **Percentage Markers**: Help track progress through the period (0% → 100%)
                
                The stretch index reveals tactical phases and team behavior patterns:
                - High values often correspond to attacking phases or transitions
                - Low values typically indicate defensive organization or pressing
                """)
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
        
        # Show centroid movement with enhanced controls AND player positions
        st.subheader("Team Centroid Movement & Player Positions")
        
        # Add playback speed control and player display toggle
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            playback_speed = st.slider(
                "Playback Speed", 
                min_value=0.1, 
                max_value=10.0, 
                value=2.0, 
                step=0.1,
                help="1.0 = real-time, 2.0 = twice as fast, etc."
            )
        
        with col2:
            show_players = st.checkbox("Show Player Positions", value=True,
                                     help="Display individual player positions alongside the centroid")
        
        with col3:
            st.markdown(f"""
            **Animation**  
            {playback_speed}x speed  
            ({period_info['duration_minutes']/playback_speed:.1f} min)
            """)
        
        # Pass tracking data to the analyzer for player positions
        if not hasattr(analyzer, 'tracking_data'):
            analyzer.tracking_data = team_manager.tracking_data
        
        # Use the enhanced centroid plot function with player positions
        centroid_plot = analyzer.create_animated_centroid_plot(
            selected_period,
            playback_speed=playback_speed,
            include_players=show_players
        )
        
        if centroid_plot:
            st.plotly_chart(centroid_plot, use_container_width=True)
            
            # Add explanation of centroid movement and player positions
            with st.expander("About Centroid & Player Positions"):
                st.markdown(f"""
                **Understanding the Visualization:**
                
                This enhanced visualization shows:
                
                1. **Individual Player Positions** - Color-coded by position (Goalkeeper, Defender, Midfielder, Forward)
                
                2. **Team Centroid** - The geometric center of all players, color-coded by Stretch Index value
                
                3. **Stretch Index Value** - The average distance of players from the centroid
                
                4. **Confidence Indicator** - Shows the reliability of the centroid and stretch index calculations
                
                **Color Coding:**
                - **Players**: Colored by position category
                - **Centroid**: Color indicates Stretch Index (blue = compact, yellow = spread out)
                
                **What to Look For:**
                - **Outliers**: Players far from the main group can distort the centroid position
                - **Formation Changes**: See how the team shape changes between defense and attack
                - **Tactical Patterns**: Identify pressing, attacking, and transition phases
                - **Anomalies**: Look for moments when the centroid position doesn't make sense
                
                **Animation Details:**
                - Real-time duration: {period_info['duration_minutes']:.1f} minutes  
                - Playback speed: {playback_speed}x  
                - Animation duration: {period_info['duration_minutes']/playback_speed:.1f} minutes
                """)
        else:
            st.warning("Insufficient data for centroid visualization.")
        
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
        - Identifying transitions between attacking and defensive phases
        - Evaluating tactical coherence and team organization
        
        **Potential Issues:**
        - Outlier players can significantly distort the stretch index
        - Goalkeepers should typically be excluded from calculations
        - Missing or erroneous player tracking data can lead to unrealistic values
        - Team tactics may not always align with expected stretch index patterns
        """)