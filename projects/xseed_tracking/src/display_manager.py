import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
import logging
from src.comparison_processor import ComparisonProcessor
import traceback
from datetime import datetime, timedelta, time
from dataclasses import dataclass

@dataclass
class TrackingData:
    """Container for different types of tracking data"""
    ground_truth: pd.DataFrame
    gpexe: pd.DataFrame
    xseed: pd.DataFrame
    protocol_info: Dict

class DisplayManager:
    """
    Handles all visualization and display functionality for the tracking analysis.
    Manages Streamlit interface and plot generation.
    """
    
    def __init__(self, data_processor: DataProcessor, comparison_processor: ComparisonProcessor):
        self.data_processor = data_processor
        self.comparison_processor = comparison_processor
        self.pitch_data = data_processor.load_pitch_data()
        
        # Define distinct colors for different elements
        self.colors = {
            'pitch_border': 'white',
            'ground_truth': 'red',
        }
        
        # Define a color mapping for any player number
        player_color_map = {
            '1': {'gpexe': 'blue', 'xseed': 'lightblue', 'ground_truth': 'red'},
            '2': {'gpexe': 'green', 'xseed': 'lightgreen', 'ground_truth': 'red'},
            '3': {'gpexe': 'purple', 'xseed': 'plum', 'ground_truth': 'red'},
            '4': {'gpexe': 'orange', 'xseed': 'gold', 'ground_truth': 'red'},
            '5': {'gpexe': 'cyan', 'xseed': 'azure', 'ground_truth': 'red'},
            '6': {'gpexe': 'magenta', 'xseed': 'pink', 'ground_truth': 'red'}
        }
        
        # Add available players to colors dictionary
        for player in data_processor.get_players_list():
            player_num = player.split()[-1]  # Extract player number
            self.colors[player] = player_color_map.get(
                player_num, 
                {'gpexe': 'gray', 'xseed': 'silver', 'ground_truth': 'red'}
            )

    
    def calculate_comparison_metrics(self, gt_data: pd.DataFrame, tracking_data: pd.DataFrame) -> Dict:
        if gt_data.empty or tracking_data.empty:
            return self._empty_metrics()

        try:
            metrics = self.comparison_processor.calculate_tracking_accuracy(
                gt_data, tracking_data
            )
            return metrics

        except Exception as e:
            logging.error(f"Error calculating comparison metrics: {str(e)}")
            logging.error(traceback.format_exc())
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary for error cases."""
        return {
            'mean_error': 0.0,
            'rmse': 0.0,
            'max_error': 0.0,
            'matching_rate': 0.0,
            'error_distribution': []
        }
    
    def display_tracking_comparison(self, ground_truth: pd.DataFrame, tracking_data: pd.DataFrame, metrics: Dict, system_type: str):
        """Display tracking comparison with enhanced metrics visualization."""
        # Extract hardware type info from the data if available
        hardware_type = tracking_data.get('hardware_type', 'old').iloc[0] if not tracking_data.empty else 'old'
        
        # Format system type with hardware type
        display_system_type = system_type
        if "xseed" in system_type.lower() and "hardware" not in system_type.lower():
            display_system_type = f"{system_type} ({hardware_type} hardware)"
        
        st.markdown(f"**{display_system_type} vs Ground Truth Comparison**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Position Traces**")
            if tracking_data.empty:
                st.warning(f"No {display_system_type} data available")
            else:
                fig = go.Figure()

                # Ground truth track - using session state
                if st.session_state.show_gt:
                    fig.add_trace(go.Scatter(
                        x=ground_truth['x_real_m'],
                        y=ground_truth['y_real_m'],
                        mode='lines',
                        name='Ground Truth',
                        line=dict(color='white', width=2)
                    ))

                # Tracking system track - using session state
                show_track = (st.session_state.show_gpexe if "gpexe" in system_type.lower() 
                            else st.session_state.show_xseed)
                if show_track:
                    # Set color based on hardware type for Xseed
                    line_color = 'blue'
                    if "xseed" in system_type.lower():
                        line_color = 'green' if hardware_type == 'old' else 'teal'
                    
                    fig.add_trace(go.Scatter(
                        x=tracking_data['x_real_m'],
                        y=tracking_data['y_real_m'],
                        mode='markers+lines',
                        name=f'{display_system_type} Track',
                        marker=dict(size=6),
                        line=dict(color=line_color, width=1)
                    ))
                # Add pitch outline if available
                if self.pitch_data is not None:
                    self.add_pitch_outline(fig)

                fig.update_layout(
                    title=f"{display_system_type} Tracking Comparison",
                    xaxis_title="X Position (m)",
                    yaxis_title="Y Position (m)",
                    template="plotly_dark",
                    showlegend=True,
                    yaxis=dict(scaleanchor="x", scaleratio=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Main error metrics section
            st.markdown("**Error Metrics**")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric("Mean Error", f"{metrics['mean_error']:.2f} m")
                st.metric("RMSE", f"{metrics['rmse']:.2f} m")
                st.metric("Median Error", f"{metrics['median_error']:.2f} m")
            
            with col2_2:
                st.metric("Min Error", f"{metrics['min_error']:.2f} m")
                st.metric("Max Error", f"{metrics['max_error']:.2f} m")
                st.metric("Matching Rate", f"{metrics['matching_rate']*100:.1f}%")
                
            # Ground truth usage metrics
            st.markdown("**Ground Truth Coverage**")
            col2_3, col2_4 = st.columns(2)
            
            with col2_3:
                st.metric("GT Points Used", f"{metrics['gt_points_used']}/{metrics['gt_points_total']}")
                st.metric("GT Coverage", f"{metrics['gt_points_usage_pct']:.1f}%")
            
            with col2_4:
                st.metric("Avg Matches per GT", f"{metrics['avg_matches_per_used_gt']:.1f}")
                st.metric("Valid Points", f"{len(metrics['error_distribution'])}")
                
            # Add skipped points information
            st.markdown("**Matching Constraints**")
            skipped_col1, skipped_col2, skipped_col3 = st.columns(3)
            
            with skipped_col1:
                st.metric("Time Window Skips", f"{metrics['skipped_time_window']}")
            
            with skipped_col2:
                st.metric("Future Time Skips", f"{metrics['skipped_future_progression']}")
                
            with skipped_col3:
                st.metric("Max Matches Skips", f"{metrics['skipped_max_matches']}")
                
            # Out of bounds points if any
            if 'out_of_bounds_count' in metrics and metrics['out_of_bounds_count'] > 0:
                st.markdown("**Out-of-Bounds Points**")
                st.metric("Count", f"{metrics['out_of_bounds_count']}")
                st.metric("Mean Error", f"{metrics['out_of_bounds_mean']:.2f} m")
            
            # Add a button to show detailed metrics
            if st.button("Show Detailed Metrics", key=f"detailed_metrics_{system_type}"):
                if 'has_interp_data' in metrics and metrics['has_interp_data']:
                    # Display enhanced metrics
                    st.markdown("### Detailed Metrics Analysis")
                    
                    # Create tabs for different metric groups
                    detailed_tabs = st.tabs(["Original Data", "Interpolated Data", "Error Distribution", "Advanced Stats"])
                    
                    with detailed_tabs[0]:
                        st.subheader("Original Data Metrics")
                        original = metrics['original']
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Mean Error", f"{original['mean_error']:.2f} m")
                            st.metric("RMSE", f"{original['rmse']:.2f} m")
                            st.metric("Median Error", f"{original['median_error']:.2f} m")
                        
                        with cols[1]:
                            st.metric("Min Error", f"{original['min_error']:.2f} m")
                            st.metric("Max Error", f"{original['max_error']:.2f} m")
                            st.metric("Points Count", f"{original['count']}")
                            st.metric("Percentage", f"{original['percentage']:.1f}%")
                    
                    with detailed_tabs[1]:
                        st.subheader("Interpolated Data Metrics")
                        interpolated = metrics['interpolated']
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Mean Error", f"{interpolated['mean_error']:.2f} m")
                            st.metric("RMSE", f"{interpolated['rmse']:.2f} m")
                            st.metric("Median Error", f"{interpolated['median_error']:.2f} m")
                        
                        with cols[1]:
                            st.metric("Min Error", f"{interpolated['min_error']:.2f} m")
                            st.metric("Max Error", f"{interpolated['max_error']:.2f} m")
                            st.metric("Points Count", f"{interpolated['count']}")
                            st.metric("Percentage", f"{interpolated['percentage']:.1f}%")
                    
                    with detailed_tabs[2]:
                        # Create error distribution histogram
                        if len(metrics['error_distribution']) > 0:
                            error_fig = go.Figure()
                            error_fig.add_trace(go.Histogram(
                                x=metrics['error_distribution'],
                                nbinsx=20,
                                marker_color=line_color,
                                opacity=0.7
                            ))
                            
                            error_fig.update_layout(
                                title="Error Distribution Histogram",
                                xaxis_title="Error (m)",
                                yaxis_title="Frequency",
                                template="plotly_dark"
                            )
                            
                            # Add lines for different statistical measures
                            error_fig.add_vline(
                                x=metrics['mean_error'], 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Mean",
                                annotation_position="top right"
                            )
                            
                            error_fig.add_vline(
                                x=metrics['median_error'], 
                                line_dash="dash", 
                                line_color="green",
                                annotation_text="Median",
                                annotation_position="top left"
                            )
                            
                            error_fig.add_vline(
                                x=metrics['min_error'], 
                                line_dash="dot", 
                                line_color="blue",
                                annotation_text="Min",
                                annotation_position="bottom left"
                            )
                            
                            st.plotly_chart(error_fig, use_container_width=True)
                            
                            # Add box plot for error distribution
                            box_fig = go.Figure()
                            box_fig.add_trace(go.Box(
                                y=metrics['error_distribution'],
                                name="Error Distribution",
                                boxpoints='all',
                                jitter=0.3,
                                pointpos=-1.8,
                                marker_color=line_color
                            ))
                            
                            box_fig.update_layout(
                                title="Error Distribution Box Plot",
                                yaxis_title="Error (m)",
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(box_fig, use_container_width=True)
                        else:
                            st.warning("No error data available for distribution visualization")
                    
                    with detailed_tabs[3]:
                        st.subheader("Advanced Statistics")
                        
                        # Format constraint statistics
                        constraint_data = pd.DataFrame({
                            'Constraint Type': ["Time Window", "Forward Time Progression", "Max Matches Per GT Point"],
                            'Points Skipped': [
                                metrics['skipped_time_window'],
                                metrics['skipped_future_progression'],
                                metrics['skipped_max_matches']
                            ],
                            'Percentage': [
                                100 * metrics['skipped_time_window'] / len(tracking_data) if len(tracking_data) > 0 else 0,
                                100 * metrics['skipped_future_progression'] / len(tracking_data) if len(tracking_data) > 0 else 0,
                                100 * metrics['skipped_max_matches'] / len(tracking_data) if len(tracking_data) > 0 else 0
                            ]
                        })
                        
                        constraint_data['Percentage'] = constraint_data['Percentage'].apply(lambda x: f"{x:.1f}%")
                        
                        st.write("**Constraint Statistics**")
                        st.dataframe(constraint_data)
                        
                        # Calculate percentile statistics
                        if len(metrics['error_distribution']) > 0:
                            percentiles = [25, 50, 75, 90, 95, 99]
                            percentile_values = np.percentile(metrics['error_distribution'], percentiles)
                            
                            percentile_data = pd.DataFrame({
                                'Percentile': [f"{p}th" for p in percentiles],
                                'Error (m)': [f"{val:.3f}" for val in percentile_values]
                            })
                            
                            st.write("**Error Percentiles**")
                            st.dataframe(percentile_data)
                            
                            # Show raw distribution as option
                            if st.checkbox("Show raw error distribution"):
                                st.write("**Raw Error Values (m)**")
                                errors_df = pd.DataFrame({'Error (m)': sorted(metrics['error_distribution'])})
                                st.dataframe(errors_df)
                else:
                    st.info("Detailed metrics about interpolated data are not available for this dataset.")
                    
                    # Show error distribution histogram for all data
                    if len(metrics['error_distribution']) > 0:
                        error_fig = go.Figure()
                        error_fig.add_trace(go.Histogram(
                            x=metrics['error_distribution'],
                            nbinsx=20,
                            marker_color=line_color if 'line_color' in locals() else 'blue',
                            opacity=0.7
                        ))
                        
                        error_fig.update_layout(
                            title="Error Distribution Histogram",
                            xaxis_title="Error (m)",
                            yaxis_title="Frequency",
                            template="plotly_dark"
                        )
                        
                        # Add statistical lines
                        error_fig.add_vline(
                            x=metrics['mean_error'], 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text="Mean",
                            annotation_position="top right"
                        )
                        
                        error_fig.add_vline(
                            x=metrics['median_error'], 
                            line_dash="dash", 
                            line_color="green",
                            annotation_text="Median",
                            annotation_position="top left"
                        )
                        
                        st.plotly_chart(error_fig, use_container_width=True)
                    
                    # Debug: Display raw metrics to help troubleshoot
                    if st.checkbox("Show raw metrics data", key=f"debug_metrics_{system_type}"):
                        st.write(metrics)

    def _create_track_visualization(self, ground_truth: pd.DataFrame, tracking_data: pd.DataFrame, system_type: str) -> go.Figure:
        """Create position trace plot comparing ground truth and tracking data."""
        fig = go.Figure()

        # Add ground truth trace
        fig.add_trace(go.Scatter(
            x=ground_truth['x_real_m'],
            y=ground_truth['y_real_m'],
            mode='lines',
            name='Ground Truth',
            line=dict(color='red', width=2)
        ))

        # Add tracking system trace with speed coloring if available
        if 'speed' in tracking_data.columns:
            fig.add_trace(go.Scatter(
                x=tracking_data['x_real_m'],
                y=tracking_data['y_real_m'],
                mode='markers+lines',
                name=f'{system_type} Track',
                marker=dict(
                    size=6,
                    color=tracking_data['speed'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Speed (m/s)",
                        x=1.2,        # Move colorbar even further right
                        titleside='right',
                        y=0.5,        # Center the colorbar vertically
                        len=0.75,     # Reduce colorbar length
                        thickness=20   # Make colorbar slightly thicker
                    )
                ),
                line=dict(color='blue' if system_type == "GPEXE" else 'green', width=1)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=tracking_data['x_real_m'],
                y=tracking_data['y_real_m'],
                mode='lines',
                name=f'{system_type} Track',
                line=dict(color='blue' if system_type == "GPEXE" else 'green', width=1)
            ))

        # Add pitch outline if available
        if self.pitch_data is not None:
            self.add_pitch_outline(fig)

        # Update layout with more space and better positioning
        fig.update_layout(
            title=f"{system_type} Tracking Comparison",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_dark",
            showlegend=True,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            # Increase right margin significantly
            margin=dict(r=200, t=50, b=50, l=50),
            # Move legend to top right, well separated from colorbar
            legend=dict(
                x=1.3,          # Move legend even further right
                y=1.0,          # Place at top
                xanchor='left',
                yanchor='top',
                orientation='v'  # Vertical orientation
            ),
            # Control the plot size
            width=800,          # Increase overall width to accommodate legend and colorbar
            height=600,         # Set a fixed height
            # Ensure the main plot maintains its size
            autosize=False
        )

        return fig
    def _create_error_distribution(self, errors: List[float]) -> go.Figure:
        """Create histogram of tracking errors."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            name='Error Distribution'
        ))
        
        fig.update_layout(
            title="Error Distribution",
            xaxis_title="Error (m)",
            yaxis_title="Frequency",
            template="plotly_dark",
            showlegend=False
        )
        
        return fig

    def display_combined_analysis(self, gt_data: pd.DataFrame, gpexe_data: pd.DataFrame, 
                        xseed_data: pd.DataFrame, combined_metrics: Dict):
    # Extract hardware type from Xseed data
        hardware_type = xseed_data.get('hardware_type', 'old').iloc[0] if not xseed_data.empty else 'old'
        
        st.subheader(f"Combined Analysis ({hardware_type.capitalize()} Hardware)")
        
        # Original ground truth comparisons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### GPEXE vs Ground Truth")
            gpexe_metrics = combined_metrics.get('gpexe', {})
            st.metric("Mean Error", f"{gpexe_metrics.get('mean_error', 0):.2f} m")
            st.metric("RMSE", f"{gpexe_metrics.get('rmse', 0):.2f} m")
            st.metric("Coverage", f"{gpexe_metrics.get('matching_rate', 0)*100:.1f}%")

        with col2:
            st.markdown(f"### Xseed ({hardware_type}) vs Ground Truth")
            xseed_metrics = combined_metrics.get('xseed', {})
            st.metric("Mean Error", f"{xseed_metrics.get('mean_error', 0):.2f} m")
            st.metric("RMSE", f"{xseed_metrics.get('rmse', 0):.2f} m")
            st.metric("Coverage", f"{xseed_metrics.get('matching_rate', 0)*100:.1f}%")

        # New section for GPEXE vs Xseed comparison
        st.markdown(f"### GPEXE vs Xseed ({hardware_type}) Comparison")
        if not gpexe_data.empty and not xseed_data.empty:
            # Calculate errors using GPEXE as reference
            gpexe_xseed_metrics = self.calculate_comparison_metrics(gpexe_data, xseed_data)
            
            st.metric("Mean Difference", f"{gpexe_xseed_metrics['mean_error']:.2f} m")
            st.metric("RMSE Difference", f"{gpexe_xseed_metrics['rmse']:.2f} m")
            st.metric("Max Difference", f"{gpexe_xseed_metrics['max_error']:.2f} m")
            st.metric("Matching Rate", f"{gpexe_xseed_metrics['matching_rate']*100:.1f}%")

            # Visualization of all three systems
            fig = self._create_combined_visualization(gt_data, gpexe_data, xseed_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Both GPEXE and Xseed ({hardware_type} hardware) data required for comparison")

    def display_multi_player_tracking(self, all_data: Dict, show_gpexe: bool, show_xseed: bool, hardware_type: str = 'old') -> None:
        """
        Display multi-player tracking visualization with hardware type support.
        
        Args:
            all_data: Dictionary containing data for each player
            show_gpexe: Whether to display GPEXE tracks
            show_xseed: Whether to display Xseed tracks
            hardware_type: Hardware type to display ('old' or 'new')
        """
        fig = go.Figure()
        
        # Add tracks for each player
        for player, data in all_data.items():
            # Add ground truth track
            if 'ground_truth' in data and not data['ground_truth'].empty:
                fig.add_trace(go.Scatter(
                    x=data['ground_truth']['x_real_m'],
                    y=data['ground_truth']['y_real_m'],
                    mode='lines',
                    name=f'{player} Ground Truth',
                    line=dict(
                        color=self.colors[player]['ground_truth'],
                        width=2,
                        dash='dot'
                    )
                ))
            
            # Add GPEXE track if enabled
            if show_gpexe and 'gpexe' in data and not data['gpexe'].empty:
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
                    line=dict(color=self.colors[player]['gpexe'], width=1)
                ))
            
            # Add Xseed track if enabled
            if show_xseed and 'xseed' in data and not data['xseed'].empty:
                # Set line color based on hardware type
                line_color = self.colors[player]['xseed']
                if hardware_type == 'new':
                    # Use a slightly different shade for new hardware
                    if line_color == 'lightblue': line_color = 'deepskyblue'
                    elif line_color == 'lightgreen': line_color = 'mediumseagreen'
                    elif line_color == 'plum': line_color = 'mediumpurple'
                    elif line_color == 'gold': line_color = 'goldenrod'
                    elif line_color == 'azure': line_color = 'lightcyan'
                    elif line_color == 'pink': line_color = 'hotpink'
                
                fig.add_trace(go.Scatter(
                    x=data['xseed']['x_real_m'],
                    y=data['xseed']['y_real_m'],
                    mode='markers+lines',
                    name=f'{player} Xseed ({hardware_type})',
                    marker=dict(
                        size=6,
                        color=data['xseed'].get('speed', 0),
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Speed (m/s)")
                    ),
                    line=dict(color=line_color, width=1)
                ))

        # Add pitch outline
        if self.pitch_data is not None:
            self.add_pitch_outline(fig)

        # Update layout
        fig.update_layout(
            title=f"Multi-Player Tracking Comparison ({hardware_type.capitalize()} Hardware)",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_dark",
            showlegend=True,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig, use_container_width=True)

    def display_distance_analysis(self, all_data: Dict):
        """Display inter-player distance analysis."""
        st.subheader("Inter-Player Distance Analysis")
        
        # Calculate distances between each pair of players
        distances = {}
        players = list(all_data.keys())
        
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pair = f"{players[i]}-{players[j]}"
                distances[pair] = self.calculate_player_distances(
                    all_data[players[i]]['ground_truth'],
                    all_data[players[j]]['ground_truth']
                )
        
        self.display_distance_metrics(distances)

    def display_distance_metrics_comparison(self, 
                                gt_distances: Dict[str, Dict], 
                                gpexe_distances: Optional[Dict[str, Dict]], 
                                xseed_distances: Optional[Dict[str, Dict]],
                                hardware_type: str = 'old'):
        """Display comparison of inter-player distances for different tracking systems with hardware type support."""
        
        for pair_name in gt_distances.keys():
            st.subheader(f"Distance Analysis: {pair_name} ({hardware_type.capitalize()} Hardware)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Get common time range
                gt_seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
                                    for t in gt_distances[pair_name]['timestamps']])
                
                if gpexe_distances and pair_name in gpexe_distances:
                    gpexe_seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
                                            for t in gpexe_distances[pair_name]['timestamps']])
                else:
                    gpexe_seconds = np.array([])
                    
                if xseed_distances and pair_name in xseed_distances:
                    xseed_seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
                                            for t in xseed_distances[pair_name]['timestamps']])
                else:
                    xseed_seconds = np.array([])
                
                # Find common time range
                min_time = min(np.min(gt_seconds), 
                            np.min(gpexe_seconds) if gpexe_seconds.size > 0 else np.inf,
                            np.min(xseed_seconds) if xseed_seconds.size > 0 else np.inf)
                max_time = max(np.max(gt_seconds),
                            np.max(gpexe_seconds) if gpexe_seconds.size > 0 else -np.inf,
                            np.max(xseed_seconds) if xseed_seconds.size > 0 else -np.inf)
                
                # Create uniform time base
                time_base = np.linspace(min_time, max_time, num=int((max_time-min_time)*10))  # 10Hz sampling
                
                # Interpolate all signals to common time base
                gt_distances_interp = np.interp(time_base, gt_seconds, gt_distances[pair_name]['distances'])
                
                if gpexe_seconds.size > 0:
                    gpexe_distances_interp = np.interp(time_base, gpexe_seconds, gpexe_distances[pair_name]['distances'])
                
                if xseed_seconds.size > 0:
                    xseed_distances_interp = np.interp(time_base, xseed_seconds, xseed_distances[pair_name]['distances'])
                
                # Convert time_base back to datetime for plotting
                plot_times = [datetime.combine(datetime.now().date(), 
                            time(hour=int(t//3600),
                                minute=int((t%3600)//60),
                                second=int(t%60),
                                microsecond=int((t%1)*1e6)))
                            for t in time_base]
                
                # Create distance plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=plot_times,
                    y=gt_distances_interp,
                    mode='lines',
                    name='Ground Truth',
                    line=dict(color='red', width=2)
                ))
                
                if gpexe_seconds.size > 0:
                    fig.add_trace(go.Scatter(
                        x=plot_times,
                        y=gpexe_distances_interp,
                        mode='lines',
                        name='GPEXE',
                        line=dict(color='blue', width=2)
                    ))
                
                if xseed_seconds.size > 0:
                    # Use different color based on hardware type
                    line_color = 'green' if hardware_type == 'old' else 'teal'
                    
                    fig.add_trace(go.Scatter(
                        x=plot_times,
                        y=xseed_distances_interp,
                        mode='lines',
                        name=f'Xseed ({hardware_type})',
                        line=dict(color=line_color, width=2)
                    ))
                
                fig.update_layout(
                    title=f"Inter-player Distance Over Time ({hardware_type.capitalize()} Hardware)",
                    xaxis_title="Time",
                    yaxis_title="Distance (m)",
                    template="plotly_dark",
                    height=400,
                    xaxis=dict(
                        tickformat='%H:%M:%S.%L',
                        tickmode='auto',
                        nticks=10
                    ),
                    yaxis=dict(
                        range=[0, max(max(gt_distances_interp), 
                                    max(gpexe_distances_interp) if gpexe_seconds.size > 0 else 0,
                                    max(xseed_distances_interp) if xseed_seconds.size > 0 else 0) * 1.1]
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"distance_{pair_name}")
            
            with col2:
                # Calculate and plot errors
                fig_error = go.Figure()
                
                if gpexe_seconds.size > 0:
                    gpexe_errors = np.abs(gpexe_distances_interp - gt_distances_interp)
                    fig_error.add_trace(go.Scatter(
                        x=plot_times,
                        y=gpexe_errors,
                        mode='lines',
                        name='GPEXE Error',
                        line=dict(color='blue', width=2)
                    ))
                
                if xseed_seconds.size > 0:
                    xseed_errors = np.abs(xseed_distances_interp - gt_distances_interp)
                    
                    # Use different color based on hardware type
                    line_color = 'green' if hardware_type == 'old' else 'teal'
                    
                    fig_error.add_trace(go.Scatter(
                        x=plot_times,
                        y=xseed_errors,
                        mode='lines',
                        name=f'Xseed Error ({hardware_type})',
                        line=dict(color=line_color, width=2)
                    ))
                
                fig_error.update_layout(
                    title=f"Distance Error Over Time ({hardware_type.capitalize()} Hardware)",
                    xaxis_title="Time",
                    yaxis_title="Error (m)",
                    template="plotly_dark",
                    height=400,
                    xaxis=dict(
                        tickformat='%H:%M:%S.%L',
                        tickmode='auto',
                        nticks=10
                    ),
                    yaxis=dict(
                        range=[0, max(max(gpexe_errors) if gpexe_seconds.size > 0 else 0,
                                    max(xseed_errors) if xseed_seconds.size > 0 else 0) * 1.1]
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(fig_error, use_container_width=True, key=f"error_{pair_name}")
                
                # Display error statistics
                st.markdown("### Error Statistics")
                cols = st.columns(2)
                
                if gpexe_seconds.size > 0:
                    with cols[0]:
                        st.write("GPEXE:")
                        st.metric("Mean Error", f"{np.mean(gpexe_errors):.2f} m")
                        st.metric("Max Error", f"{np.max(gpexe_errors):.2f} m")
                        st.metric("RMSE", f"{np.sqrt(np.mean(np.square(gpexe_errors))):.2f} m")
                
                if xseed_seconds.size > 0:
                    with cols[1]:
                        st.write(f"Xseed ({hardware_type}):")
                        st.metric("Mean Error", f"{np.mean(xseed_errors):.2f} m")
                        st.metric("Max Error", f"{np.max(xseed_errors):.2f} m")
                        st.metric("RMSE", f"{np.sqrt(np.mean(np.square(xseed_errors))):.2f} m")
            
            st.markdown("---")

    def calculate_player_distances(self, data1: pd.DataFrame, data2: pd.DataFrame) -> List[float]:
        """Calculate distances between two players' tracks."""
        # Find common timestamps
        common_times = pd.Series(list(set(data1['date time']) & set(data2['date time'])))
        distances = []
        
        if not common_times.empty:
            data1_filtered = data1[data1['date time'].isin(common_times)]
            data2_filtered = data2[data2['date time'].isin(common_times)]
            
            distances = np.sqrt(
                (data1_filtered['x_real_m'].values - data2_filtered['x_real_m'].values)**2 +
                (data1_filtered['y_real_m'].values - data2_filtered['y_real_m'].values)**2
            )
        
        return distances.tolist() if isinstance(distances, np.ndarray) else []
    
    def display_speed_analysis(self, tracking_data: pd.DataFrame, protocol_info: Dict):
        """
        Display speed analysis including timeline and statistics.
        
        Args:
            tracking_data: DataFrame containing tracking data with speed information
            protocol_info: Dictionary containing protocol details
        """
        st.subheader("Speed Analysis")
        
        if tracking_data.empty:
            st.warning("No tracking data available")
            return

        # Speed timeline plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tracking_data['date time'],
            y=tracking_data['speed'],
            name='Speed',
            mode='lines',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=f"Speed Timeline - {protocol_info.get('Protocol name', 'Unknown Protocol')}",
            xaxis_title="Time",
            yaxis_title="Speed (m/s)",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Speed statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Speed Stats")
            st.metric("Max Speed", f"{tracking_data['speed'].max():.2f} m/s")
            st.metric("Average Speed", f"{tracking_data['speed'].mean():.2f} m/s")
            st.metric("Total Distance", 
                     f"{tracking_data['gps_distance_from_previous'].sum():.2f} m")

        with col2:
            if 'gps_speed_status' in tracking_data.columns:
                st.write("### Movement Analysis")
                walking = len(tracking_data[tracking_data['gps_speed_status'] == 'Walking'])
                running = len(tracking_data[tracking_data['gps_speed_status'] == 'Running'])
                st.metric("Walking Time", f"{walking/10:.1f} s")
                st.metric("Running Time", f"{running/10:.1f} s")

    def display_system_comparison(self, ground_truth: pd.DataFrame, 
                        tracking_data: pd.DataFrame, 
                        xseed_data: pd.DataFrame, 
                        metrics: Dict, 
                        protocol_info: Dict, 
                        selected_player: str):
        """Display comprehensive comparison between tracking systems with hardware type support."""
        # Extract hardware type
        hardware_type = 'old'
        if not xseed_data.empty and 'hardware_type' in xseed_data.columns:
            hardware_type = xseed_data['hardware_type'].iloc[0]
        
        st.subheader(f"System Comparison Analysis - {hardware_type.capitalize()} Hardware")

        # Display metrics comparison
        col1, col2 = st.columns(2)
        with col1:
            gpexe_metrics = metrics.get('gpexe', {})
            st.write("### GPEXE System")
            st.metric("Mean Error", f"{gpexe_metrics.get('mean_error', 0):.2f} m")
            st.metric("RMSE", f"{gpexe_metrics.get('rmse', 0):.2f} m")
            st.metric("Max Error", f"{gpexe_metrics.get('max_error', 0):.2f} m")
            st.metric("Coverage", f"{gpexe_metrics.get('matching_rate', 0)*100:.1f}%")
            
            if 'velocity' in gpexe_metrics:
                st.metric("Max Speed", f"{gpexe_metrics['velocity']['max_speed']:.2f} m/s")

        with col2:
            xseed_metrics = metrics.get('xseed', {})
            st.write(f"### Xseed System ({hardware_type})")
            st.metric("Mean Error", f"{xseed_metrics.get('mean_error', 0):.2f} m")
            st.metric("RMSE", f"{xseed_metrics.get('rmse', 0):.2f} m")
            st.metric("Max Error", f"{xseed_metrics.get('max_error', 0):.2f} m")
            st.metric("Coverage", f"{xseed_metrics.get('matching_rate', 0)*100:.1f}%")
            
            if 'velocity' in xseed_metrics:
                st.metric("Max Speed", f"{xseed_metrics['velocity']['max_speed']:.2f} m/s")

        # Create visualization tabs
        tabs = st.tabs(["Position Tracks", "Error Distribution", "Speed Comparison"])
        
        with tabs[0]:
            fig = go.Figure()

            # Ground truth track
            if not ground_truth.empty:
                ground_truth['x_real_m'] = pd.to_numeric(ground_truth['x_real_m'], errors='coerce')
                ground_truth['y_real_m'] = pd.to_numeric(ground_truth['y_real_m'], errors='coerce')
                fig.add_trace(go.Scatter(
                    x=ground_truth['x_real_m'],
                    y=ground_truth['y_real_m'],
                    mode='lines',
                    name='Ground Truth',
                    line=dict(color='white', width=2)
                ))

            # GPEXE track
            if not tracking_data.empty:
                tracking_data['x_real_m'] = pd.to_numeric(tracking_data['x_real_m'], errors='coerce')
                tracking_data['y_real_m'] = pd.to_numeric(tracking_data['y_real_m'], errors='coerce')
                fig.add_trace(go.Scatter(
                    x=tracking_data['x_real_m'],
                    y=tracking_data['y_real_m'],
                    mode='markers+lines',
                    name='GPEXE Track',
                    marker=dict(
                        size=6,
                        color=tracking_data.get('speed', 0),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Speed (m/s)")
                    ),
                    line=dict(color='blue', width=1)
                ))

            # Xseed track
            if not xseed_data.empty:
                xseed_data['x_real_m'] = pd.to_numeric(xseed_data['x_real_m'], errors='coerce')
                xseed_data['y_real_m'] = pd.to_numeric(xseed_data['y_real_m'], errors='coerce')
                
                # Set line color based on hardware type
                line_color = 'green' if hardware_type == 'old' else 'teal'
                
                fig.add_trace(go.Scatter(
                    x=xseed_data['x_real_m'],
                    y=xseed_data['y_real_m'],
                    mode='markers+lines',
                    name=f'Xseed Track ({hardware_type})',
                    marker=dict(size=6),
                    line=dict(color=line_color, width=1)
                ))

            # Add pitch outline and update layout
            self.add_pitch_outline(fig)
            fig.update_layout(
                title=f"Combined Position Tracks ({hardware_type.capitalize()} Hardware)",
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)",
                template="plotly_dark",
                showlegend=True,
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
                
        with tabs[1]:
            self._plot_error_distributions(
                gpexe_metrics.get('error_distribution', []),
                xseed_metrics.get('error_distribution', []),
                hardware_type
            )
                
        with tabs[2]:
            self._plot_speed_comparison(tracking_data, xseed_data, hardware_type)

    def _plot_combined_tracks(self, ground_truth: pd.DataFrame, tracking_data: pd.DataFrame, xseed_data: pd.DataFrame):
        fig = go.Figure()

        # Ground truth track
        if not ground_truth.empty:
            ground_truth['x_real_m'] = pd.to_numeric(ground_truth['x_real_m'], errors='coerce')
            ground_truth['y_real_m'] = pd.to_numeric(ground_truth['y_real_m'], errors='coerce')
            fig.add_trace(go.Scatter(
                x=ground_truth['x_real_m'],
                y=ground_truth['y_real_m'],
                mode='lines',
                name='Ground Truth',
                line=dict(color='white', width=2)
            ))

        # GPEXE track
        if not tracking_data.empty:
            tracking_data['x_real_m'] = pd.to_numeric(tracking_data['x_real_m'], errors='coerce')
            tracking_data['y_real_m'] = pd.to_numeric(tracking_data['y_real_m'], errors='coerce')
            fig.add_trace(go.Scatter(
                x=tracking_data['x_real_m'],
                y=tracking_data['y_real_m'],
                mode='markers+lines',
                name='GPEXE Track',
                marker=dict(
                    size=6,
                    color=tracking_data.get('speed', 0),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed (m/s)")
                ),
                line=dict(color='blue', width=1)
            ))

        # Xseed track
        if not xseed_data.empty:
            xseed_data['x_real_m'] = pd.to_numeric(xseed_data['x_real_m'], errors='coerce')
            xseed_data['y_real_m'] = pd.to_numeric(xseed_data['y_real_m'], errors='coerce')
            fig.add_trace(go.Scatter(
                x=xseed_data['x_real_m'],
                y=xseed_data['y_real_m'],
                mode='markers+lines',
                name='Xseed Track',
                marker=dict(size=6),
                line=dict(color='green', width=1)
            ))

        # Add pitch outline and update layout
        self.add_pitch_outline(fig)
        fig.update_layout(
            title="Combined Position Tracks",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_dark",
            showlegend=True,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def _plot_error_distributions(self, gpexe_errors: List[float], xseed_errors: List[float], hardware_type: str = 'old'):
        """Plot error distributions for both tracking systems."""
        fig = go.Figure()
        
        if gpexe_errors:
            fig.add_trace(go.Histogram(
                x=gpexe_errors,
                name='GPEXE Errors',
                opacity=0.75,
                nbinsx=30
            ))
        
        if xseed_errors:
            fig.add_trace(go.Histogram(
                x=xseed_errors,
                name=f'Xseed Errors ({hardware_type})',
                opacity=0.75,
                nbinsx=30
            ))
        
        fig.update_layout(
            title=f"Error Distribution Comparison ({hardware_type.capitalize()} Hardware)",
            xaxis_title="Error (m)",
            yaxis_title="Frequency",
            template="plotly_dark",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_speed_comparison(self, tracking_data: pd.DataFrame, xseed_data: pd.DataFrame, hardware_type: str = 'old'):
        """Create speed comparison visualization with hardware type support."""
        fig = go.Figure()

        # Add GPEXE speed trace
        if not tracking_data.empty and 'speed' in tracking_data.columns:
            fig.add_trace(go.Scatter(
                x=tracking_data['date time'],
                y=tracking_data['speed'],
                name='GPEXE Speed',
                mode='lines',
                line=dict(color='blue', width=2)
            ))

        # Add Xseed speed trace
        if not xseed_data.empty and 'speed' in xseed_data.columns:
            # Different color for different hardware
            line_color = 'green' if hardware_type == 'old' else 'teal'
            
            fig.add_trace(go.Scatter(
                x=xseed_data['date time'],
                y=xseed_data['speed'],
                name=f'Xseed Speed ({hardware_type})',
                mode='lines',
                line=dict(color=line_color, width=2)
            ))

        fig.update_layout(
            title=f"Speed Comparison ({hardware_type.capitalize()} Hardware)",
            xaxis_title="Time",
            yaxis_title="Speed (m/s)",
            template="plotly_dark",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    def add_pitch_outline(self, fig: go.Figure):
        """Add pitch outline to figure."""
        if self.pitch_data is not None:
            # Add main outline
            fig.add_trace(go.Scatter(
                x=self.pitch_data['x_real_m'],
                y=self.pitch_data['y_real_m'],
                mode='lines',
                name='Pitch',
                line=dict(color=self.colors['pitch_border'], width=2),
                showlegend=False
            ))
            
            # Close the pitch border
            fig.add_trace(go.Scatter(
                x=[self.pitch_data['x_real_m'].iloc[-1], 
                self.pitch_data['x_real_m'].iloc[0]],
                y=[self.pitch_data['y_real_m'].iloc[-1], 
                self.pitch_data['y_real_m'].iloc[0]],
                mode='lines',
                line=dict(color=self.colors['pitch_border'], width=2),
                showlegend=False
            ))

    def display_protocol_info(self, protocol_info: Dict):
        """Display protocol information in a formatted way."""
        st.write("### Protocol Information")
        info_dict = {
            "Protocol ID": protocol_info.get('Protocol ID', 'N/A'),
            "Protocol Name": protocol_info.get('Protocol name', 'N/A'),
            "Start Time": str(protocol_info.get('Start_Time', 'N/A')),
            "End Time": str(protocol_info.get('End_Time', 'N/A')),
            "Multi Analysis": protocol_info.get('multi_analysis', False)
        }
        st.json(info_dict)

    def display_coverage_metrics(self, metrics: Dict):
        """Display coverage metrics."""
        st.write("### Coverage Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Temporal Coverage", 
                     f"{metrics.get('temporal_coverage', 0)*100:.1f}%")
            st.metric("Total Time", 
                     f"{metrics.get('total_time', 0):.1f} s")
        
        with col2:
            st.metric("Spatial Coverage", 
                     f"{metrics.get('spatial_coverage', 0)*100:.1f}%")
            st.metric("Covered Time", 
                     f"{metrics.get('covered_time', 0):.1f} s")

    def run_multi_player_view(self):
        """Handle multi-player comparison view."""
        protocol_df = self.data_processor.get_protocol_info('Player 1')
        multi_protocols = protocol_df[protocol_df['multi_analysis'] == True]
        
        if multi_protocols.empty:
            st.warning("No multi-player protocols available")
            return
        
        # Protocol selection
        protocol_id = st.sidebar.selectbox(
            "Select Protocol",
            multi_protocols['Protocol ID'].unique(),
            format_func=lambda x: f"Protocol {x}: {multi_protocols[multi_protocols['Protocol ID'] == x]['Protocol name'].iloc[0]}"
        )
        
        # Display controls
        show_gpexe = st.sidebar.checkbox("Show GPEXE Tracks", value=True)
        show_xseed = st.sidebar.checkbox("Show Xseed Tracks", value=True)
        show_distances = st.sidebar.checkbox("Show Inter-Player Distances", value=True)

        try:
            # Process data for all players
            all_data = self._process_multi_player_data(protocol_id)
            
            if all_data:
                tabs = st.tabs(["Combined View", "Distance Analysis", "Speed Analysis"])
                
                with tabs[0]:
                    self.display_multi_player_tracking(
                        all_data,
                        show_gpexe,
                        show_xseed
                    )
                
                with tabs[1]:
                    if show_distances:
                        self.display_distance_analysis(all_data)
                
                with tabs[2]:
                    self._display_multi_player_speed_analysis(all_data)
            else:
                st.warning("No data available for selected protocol")
                
        except Exception as e:
            st.error(f"Error in multi-player analysis: {str(e)}")
            logging.error(f"Multi-player analysis error: {str(e)}", exc_info=True)

    def _process_multi_player_data(self, protocol_id: int) -> Dict:
        """Process data for all players in a multi-player protocol."""
        all_data = {}
        for player in self.data_processor.get_players_list():
            gt_data, tracking_data, protocol_info = self.data_processor.process_player_data(
                player, protocol_id, 2  # Default time tolerance
            )
            
            if not gt_data.empty:
                all_data[player] = {
                    'ground_truth': gt_data,
                    'gpexe': tracking_data,
                    'protocol_info': protocol_info
                }
                
                # Load Xseed data
                xseed_data = self.data_processor.load_xseed_data(
                    player,
                    protocol_info['Start_Time'],
                    protocol_info['End_Time']
                )
                if not xseed_data.empty:
                    all_data[player]['xseed'] = xseed_data
        
        return all_data

    def _display_multi_player_speed_analysis(self, all_data: Dict):
        """Display speed analysis for multiple players."""
        for player, data in all_data.items():
            st.write(f"### {player} Speed Analysis")
            if 'gpexe' in data and 'speed' in data['gpexe'].columns:
                self.display_speed_analysis(
                    data['gpexe'],
                    data['protocol_info']
                )
            else:
                st.warning(f"No speed data available for {player}")

    
    def _calculate_errors(self, gt_data: Dict, track_data: Dict) -> np.ndarray:
        """Calculate errors by interpolating ground truth to tracking timestamps."""
        if not gt_data['timestamps'] or not track_data['timestamps']:
            return np.array([])
        
        # Convert times to seconds for interpolation
        gt_seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
                            for t in gt_data['timestamps']])
        track_seconds = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
                                for t in track_data['timestamps']])
        
        # Interpolate ground truth distances to tracking timestamps
        gt_interp = np.interp(track_seconds, gt_seconds, gt_data['distances'])
        
        # Calculate absolute errors
        errors = np.abs(np.array(track_data['distances']) - gt_interp)
        return errors
    def _calculate_synchronized_errors(self, gt_data: Dict, track_data: Dict) -> Dict:
        """Calculate errors with properly synchronized timestamps."""
        if not gt_data['timestamps'] or not track_data['timestamps']:
            return {'timestamps': [], 'errors': []}
        
        # Convert times to seconds for interpolation
        gt_seconds = np.array([
            t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
            for t in gt_data['timestamps']
        ])
        track_seconds = np.array([
            t.hour * 3600 + t.minute * 60 + t.second + t.microsecond/1e6 
            for t in track_data['timestamps']
        ])
        
        # Get overlap range
        min_time = max(min(gt_seconds), min(track_seconds))
        max_time = min(max(gt_seconds), max(track_seconds))
        
        # Create uniform time points in overlap range
        uniform_times = np.linspace(min_time, max_time, num=int((max_time-min_time)*10))  # 10Hz
        
        # Interpolate both signals
        gt_interp = np.interp(uniform_times, gt_seconds, gt_data['distances'])
        track_interp = np.interp(uniform_times, track_seconds, track_data['distances'])
        
        # Calculate errors
        errors = np.abs(track_interp - gt_interp)
        
        # Convert uniform times back to time objects
        timestamps = [
            datetime.combine(datetime.now().date(), 
                            time(hour=int(t//3600),
                                minute=int((t%3600)//60),
                                second=int(t%60),
                                microsecond=int((t%1)*1e6)))
            for t in uniform_times
        ]
        
        return {
            'timestamps': timestamps,
            'errors': errors.tolist()
        }
    def _create_combined_visualization(self, gt_data: pd.DataFrame, gpexe_data: pd.DataFrame, xseed_data: pd.DataFrame) -> go.Figure:
        """Create a combined visualization of ground truth and tracking data."""
        fig = go.Figure()

        # Ground truth track
        if not gt_data.empty:
            fig.add_trace(go.Scatter(
                x=gt_data['x_real_m'],
                y=gt_data['y_real_m'],
                mode='lines',
                name='Ground Truth',
                line=dict(color='red', width=2)
            ))

        # GPEXE track
        if not gpexe_data.empty:
            fig.add_trace(go.Scatter(
                x=gpexe_data['x_real_m'],
                y=gpexe_data['y_real_m'],
                mode='markers+lines',
                name='GPEXE Track',
                marker=dict(size=6),
                line=dict(color='blue', width=1)
            ))

        # Xseed track
        if not xseed_data.empty:
            fig.add_trace(go.Scatter(
                x=xseed_data['x_real_m'],
                y=xseed_data['y_real_m'],
                mode='markers+lines',
                name='Xseed Track',
                marker=dict(size=6),
                line=dict(color='green', width=1)
            ))

        # Add pitch outline
        if self.pitch_data is not None:
            self.add_pitch_outline(fig)

        fig.update_layout(
            title="Combined Tracking Comparison",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_dark",
            showlegend=True,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig