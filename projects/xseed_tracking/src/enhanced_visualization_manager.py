import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from src.haversine_processor import HaversineProcessor
import streamlit as st

@dataclass
class TrackingData:
    """Container for different types of tracking data"""
    ground_truth: pd.DataFrame
    gpexe: pd.DataFrame
    shinguard: pd.DataFrame
    protocol_info: Dict

class EnhancedVisualizationManager:
    def __init__(self, data_processor):
        """Initialize visualization manager with data processor."""
        self.data_processor = data_processor
        self.pitch_data = data_processor.load_pitch_data()
        self.colors = {
            'Player 1': {'gpexe': 'blue', 'shinguard': 'lightblue', 'ground_truth': 'red'},
            'Player 2': {'gpexe': 'green', 'shinguard': 'lightgreen', 'ground_truth': 'red'},
            'Player 3': {'gpexe': 'purple', 'shinguard': 'plum', 'ground_truth': 'red'}
        }
        self.DEFAULT_TIME_TOLERANCE = 2
    
    def process_player_data(self, 
                       player: str, 
                       protocol_id: int, 
                       time_tolerance: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Process data for a single player."""
        try:
            # Initialize ground truth processor
            gt_processor = GroundTruthProcessor(
                str(self.protocol_files[player]),
                [str(self.activity_files[player])]
            )

            # Get protocol info
            protocol_info = gt_processor.protocols_df[
                gt_processor.protocols_df['Protocol ID'] == protocol_id
            ].iloc[0]
            
            protocol_start = pd.to_datetime(f"2024-10-29 {protocol_info['Start_Time']}")
            protocol_end = pd.to_datetime(f"2024-10-29 {protocol_info['End_Time']}")

            # Load and process GPS data
            gps_df = pd.read_csv(self.input_files[player], sep=";")
            gps_df = self.data_sync.standardize_timestamps(gps_df)
            
            if 'gpexe' in str(self.input_files[player]).lower():
                gps_df = self.data_sync.downsample_gpexe(gps_df)

            # Filter GPS data by time window
            gps_mask = (
                (gps_df['date time'] >= protocol_start - pd.Timedelta(seconds=time_tolerance)) & 
                (gps_df['date time'] <= protocol_end + pd.Timedelta(seconds=time_tolerance))
            )
            filtered_gps = gps_df[gps_mask]
            filtered_gps = rename_columns(filtered_gps)
            
            # Load shinguard data if needed
            if protocol_info['multi_analysis']:
                shinguard_data = self.load_shinguard_data(player, protocol_start, protocol_end)
                if not shinguard_data.empty:
                    # Use merge_data_streams instead of synchronize_data_streams
                    filtered_gps = self.data_sync.merge_data_streams(
                        filtered_gps, 
                        shinguard_data,
                        merge_type='outer',
                        interpolate=True
                    )

            # Get ground truth path
            ground_truth_df = gt_processor.get_ground_truth_path(protocol_id)

            # Transform coordinates if data is available
            if not ground_truth_df.empty and not filtered_gps.empty:
                gt_transformed = self.transform_coordinates(ground_truth_df)
                gps_transformed = self.transform_coordinates(filtered_gps)
                return gt_transformed, gps_transformed, protocol_info
            
            return pd.DataFrame(), pd.DataFrame(), protocol_info
            
        except Exception as e:
            logging.error(f"Error processing player data: {str(e)}")
            raise

    def create_individual_comparison(self, player: str, tracking_data: TrackingData,
                                  comparison_type: str) -> go.Figure:
        """Create individual comparison plot (GPEXE or Shinguard vs Ground Truth)."""
        fig = go.Figure()
        
        # Add ground truth track
        fig.add_trace(go.Scatter(
            x=tracking_data.ground_truth['x_real_m'],
            y=tracking_data.ground_truth['y_real_m'],
            mode='lines',
            name='Ground Truth',
            line=dict(color=self.colors[player]['ground_truth'], width=2),
            hovertemplate=(
                "Ground Truth<br>"
                "X: %{x:.1f}m<br>"
                "Y: %{y:.1f}m<br>"
                "Time: %{customdata}<br>"
                "<extra></extra>"
            ),
            customdata=tracking_data.ground_truth['date time']
        ))

        # Add comparison track (GPEXE or Shinguard)
        comparison_data = tracking_data.gpexe if comparison_type == 'gpexe' else tracking_data.shinguard
        if not comparison_data.empty:
            fig.add_trace(go.Scatter(
                x=comparison_data['x_real_m'],
                y=comparison_data['y_real_m'],
                mode='markers+lines',
                name=comparison_type.upper(),
                marker=dict(
                    size=6,
                    color=comparison_data.get('speed', 0),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed (m/s)")
                ),
                line=dict(color=self.colors[player][comparison_type], width=1),
                hovertemplate=(
                    f"{comparison_type.upper()}<br>"
                    "X: %{x:.1f}m<br>"
                    "Y: %{y:.1f}m<br>"
                    "Speed: %{marker.color:.1f} m/s<br>"
                    "Time: %{customdata}<br>"
                    "<extra></extra>"
                ),
                customdata=comparison_data['date time']
            ))

        # Add pitch visualization
        if self.pitch_data is not None:
            self._add_pitch_visualization(fig)

        fig.update_layout(
            title=f"{player} - {comparison_type.upper()} vs Ground Truth - Protocol {tracking_data.protocol_info['Protocol ID']}",
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            template='plotly_dark',
            showlegend=True,
            hovermode='closest',
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def create_multi_player_visualization(self, all_tracking_data: Dict[str, TrackingData],
                                        show_gpexe: bool = True,
                                        show_shinguard: bool = True) -> go.Figure:
        """Create multi-player visualization."""
        fig = go.Figure()
        
        for player, tracking_data in all_tracking_data.items():
            # Add ground truth
            fig.add_trace(go.Scatter(
                x=tracking_data.ground_truth['x_real_m'],
                y=tracking_data.ground_truth['y_real_m'],
                mode='lines',
                name=f'{player} Ground Truth',
                line=dict(color=self.colors[player]['ground_truth'], width=2, dash='dot'),
                hovertemplate=(
                    f"{player} Ground Truth<br>"
                    "X: %{x:.1f}m<br>"
                    "Y: %{y:.1f}m<br>"
                    "Time: %{customdata}<br>"
                    "<extra></extra>"
                ),
                customdata=tracking_data.ground_truth['date time']
            ))

            if show_gpexe and not tracking_data.gpexe.empty:
                fig.add_trace(go.Scatter(
                    x=tracking_data.gpexe['x_real_m'],
                    y=tracking_data.gpexe['y_real_m'],
                    mode='markers+lines',
                    name=f'{player} GPEXE',
                    marker=dict(
                        size=6,
                        color=tracking_data.gpexe.get('speed', 0),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Speed (m/s)")
                    ),
                    line=dict(color=self.colors[player]['gpexe'], width=1),
                    hovertemplate=(
                        f"{player} GPEXE<br>"
                        "X: %{x:.1f}m<br>"
                        "Y: %{y:.1f}m<br>"
                        "Speed: %{marker.color:.1f} m/s<br>"
                        "Time: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=tracking_data.gpexe['date time']
                ))

            if show_shinguard and not tracking_data.shinguard.empty:
                fig.add_trace(go.Scatter(
                    x=tracking_data.shinguard['x_real_m'],
                    y=tracking_data.shinguard['y_real_m'],
                    mode='markers+lines',
                    name=f'{player} Shinguard',
                    marker=dict(size=6),
                    line=dict(color=self.colors[player]['shinguard'], width=1),
                    hovertemplate=(
                        f"{player} Shinguard<br>"
                        "X: %{x:.1f}m<br>"
                        "Y: %{y:.1f}m<br>"
                        "Time: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=tracking_data.shinguard['date time']
                ))

        if self.pitch_data is not None:
            self._add_pitch_visualization(fig)

        protocol_info = next(iter(all_tracking_data.values())).protocol_info
        fig.update_layout(
            title=f"Multi-Player Analysis - Protocol {protocol_info['Protocol ID']}",
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            template='plotly_dark',
            showlegend=True,
            hovermode='closest',
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def _add_pitch_visualization(self, fig: go.Figure) -> None:
        """Add pitch outline to figure."""
        fig.add_trace(go.Scatter(
            x=self.pitch_data['x_real_m'],
            y=self.pitch_data['y_real_m'],
            mode='lines',
            line=dict(color='white', width=2),
            name='Pitch',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[self.pitch_data['x_real_m'].iloc[-1], self.pitch_data['x_real_m'].iloc[0]],
            y=[self.pitch_data['y_real_m'].iloc[-1], self.pitch_data['y_real_m'].iloc[0]],
            mode='lines',
            line=dict(color='white', width=2),
            showlegend=False
        ))

    def calculate_player_metrics(self, gt_data: pd.DataFrame, 
                               tracking_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate tracking accuracy metrics."""
        return PlotVisualization.calculate_player_metrics(gt_data, tracking_data)

    def calculate_player_distances(self, all_tracking_data: Dict[str, TrackingData]) -> Dict[str, List[float]]:
        """Calculate distances between players."""
        distances = {}
        players = list(all_tracking_data.keys())
        
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                player1, player2 = players[i], players[j]
                key = f"{player1}-{player2}"
                
                p1_data = all_tracking_data[player1].ground_truth
                p2_data = all_tracking_data[player2].ground_truth
                
                common_times = pd.Series(list(set(p1_data['date time']) & set(p2_data['date time'])))
                if not common_times.empty:
                    p1_filtered = p1_data[p1_data['date time'].isin(common_times)]
                    p2_filtered = p2_data[p2_data['date time'].isin(common_times)]
                    
                    distances[key] = np.sqrt(
                        (p1_filtered['x_real_m'] - p2_filtered['x_real_m'])**2 +
                        (p1_filtered['y_real_m'] - p2_filtered['y_real_m'])**2
                    ).tolist()
        
        return distances

    def display_player_distances(self, distances: Dict[str, List[float]]) -> None:
        """Display distance analysis between players."""
        st.subheader("Inter-Player Distances")
        
        cols = st.columns(len(distances))
        for idx, (pair, dist_values) in enumerate(distances.items()):
            with cols[idx]:
                st.write(f"**{pair}**")
                st.metric("Average Distance", f"{np.mean(dist_values):.2f} m")
                st.metric("Min Distance", f"{np.min(dist_values):.2f} m")
                st.metric("Max Distance", f"{np.max(dist_values):.2f} m")

        fig = go.Figure()
        for pair, dist_values in distances.items():
            fig.add_trace(go.Box(
                y=dist_values,
                name=pair,
                boxpoints='outliers'
            ))

        fig.update_layout(
            title="Distance Distributions Between Players",
            yaxis_title="Distance (m)",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)