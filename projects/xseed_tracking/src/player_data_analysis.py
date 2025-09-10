import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import re
import os
import glob
from datetime import datetime, timedelta

class PlayerDataAnalyzer:
    """Class for analyzing player data availability across time"""
    
    def __init__(self, player_manager=None, data_folder=None):
        """
        Initialize the analyzer with either a PlayerDataManager or a data folder path
        
        Args:
            player_manager (PlayerDataManager, optional): Existing player data manager
            data_folder (str, optional): Path to folder with player data files
        """
        self.player_manager = player_manager
        self.data_folder = data_folder
        self.players_data = {}
        self.global_min_time = None
        self.global_max_time = None
        
    def load_data_if_needed(self):
        """Load data if not already provided through player manager"""
        if self.player_manager is not None:
            # Use data from existing player manager
            self.players_data = self.player_manager.players_data
            self.global_min_time = self.player_manager.global_min_time
            self.global_max_time = self.player_manager.global_max_time
            return
                
        # Otherwise load data from folder
        if self.data_folder is None:
            raise ValueError("Either player_manager or data_folder must be provided")
                
        # Group files by player ID
        player_file_groups = {}
        all_files = glob.glob(os.path.join(self.data_folder, "player_*.csv"))
        print(f"Found {len(all_files)} player tracking files")
        
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
        
        # Process each player's files
        for player_id, files in player_file_groups.items():
            player_key = f"Player {player_id}"
            combined_df = pd.DataFrame()
            
            for file in files:
                filename = os.path.basename(file)
                try:
                    # Extract session and segment IDs
                    match = re.search(r'player_(\d+)_session_(\d+)_segment_(\d+)', filename)
                    if not match:
                        print(f"Couldn't extract IDs from {filename}")
                        continue
                        
                    # Read the CSV file
                    df = pd.read_csv(file, low_memory=False)
                    
                    # Convert gps_time to datetime
                    if 'gps_time' in df.columns:
                        try:
                            df['timestamp'] = pd.to_datetime(df['gps_time'], format='mixed', errors='coerce')
                            df = df.dropna(subset=['timestamp'])
                            
                            if df.empty:
                                print(f"All timestamps in {filename} were invalid after parsing")
                                continue
                            
                            # Append to combined data
                            combined_df = pd.concat([combined_df, df], ignore_index=True)
                        except Exception as date_error:
                            print(f"Error parsing dates in {filename}: {date_error}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            
            # If we have data for this player
            if not combined_df.empty:
                # Sort by timestamp
                combined_df = combined_df.sort_values('timestamp')
                
                # Update global min and max times
                min_time = combined_df['timestamp'].min()
                max_time = combined_df['timestamp'].max()
                
                if global_min_time is None or min_time < global_min_time:
                    global_min_time = min_time
                    
                if global_max_time is None or max_time > global_max_time:
                    global_max_time = max_time
                
                # Store the data
                players_data[player_key] = {
                    'data': combined_df,
                    'id': player_id
                }
        
        self.players_data = players_data
        self.global_min_time = global_min_time
        self.global_max_time = global_max_time
    
    def analyze_data_availability(self, bin_size_seconds=5, output_file=None):
        """
        Analyze and visualize player data availability over time.
        
        Args:
            bin_size_seconds (int): Size of time bins in seconds
            output_file (str, optional): Path to save the visualization
            
        Returns:
            tuple: (fig, ax, results_df) - Figure, axes and results dataframe
        """
        # Load data if needed
        self.load_data_if_needed()
        
        if not self.players_data:
            print("No valid player data found.")
            return None, None, None
        
        # Define time bins for analysis
        total_seconds = (self.global_max_time - self.global_min_time).total_seconds()
        
        # Create bins - aim for around 300 bins or user-specified size, whichever is larger
        bin_size_seconds = max(bin_size_seconds, total_seconds / 300)
        num_bins = int(total_seconds / bin_size_seconds) + 1
        
        # Create time bins
        time_bins = [self.global_min_time + timedelta(seconds=i * bin_size_seconds) 
                    for i in range(num_bins + 1)]
        
        # Create a figure for visualization
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Analyze data presence for each player
        results = []
        
        for idx, (player_name, player_info) in enumerate(sorted(self.players_data.items())):
            presence_data = []
            
            # Get player dataframe
            if isinstance(player_info, dict) and 'data' in player_info:
                df = player_info['data']
            elif isinstance(player_info, pd.DataFrame):
                df = player_info
            else:
                continue
            
            # Check for x, y presence
            has_position_mask = df['x'].notna() & df['y'].notna()
            position_df = df[has_position_mask].copy()
            
            # For each time bin, check if there's data
            for i in range(len(time_bins) - 1):
                bin_start = time_bins[i]
                bin_end = time_bins[i+1]
                
                # Check if there's any data point in this bin
                in_bin = ((position_df['timestamp'] >= bin_start) & 
                          (position_df['timestamp'] < bin_end))
                
                has_data = in_bin.any()
                
                # Store result
                bin_mid = bin_start + (bin_end - bin_start) / 2
                presence_data.append({
                    'player': player_name,
                    'time': bin_mid,
                    'has_data': has_data
                })
            
            # Convert to DataFrame
            presence_df = pd.DataFrame(presence_data)
            
            # Plot presence data
            y_pos = idx + 1
            x_times = presence_df['time']
            
            # Plot data presence as points
            for i, (t, has_data) in enumerate(zip(presence_df['time'], presence_df['has_data'])):
                if has_data:
                    ax.plot(t, y_pos, 'o', color='green', markersize=4, alpha=0.7)
            
            # Calculate statistics
            total_bins = len(presence_df)
            data_present_bins = presence_df['has_data'].sum()
            data_percentage = (data_present_bins / total_bins) * 100 if total_bins > 0 else 0
            
            # Get player ID
            if isinstance(player_info, dict) and 'id' in player_info:
                player_id = player_info['id']
            else:
                player_id = player_name.split()[-1] if len(player_name.split()) > 1 else "Unknown"
            
            # Store results
            results.append({
                'player': player_name,
                'id': player_id,
                'data_percentage': data_percentage,
                'bins_with_data': data_present_bins,
                'total_bins': total_bins
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('data_percentage', ascending=False)
        
        # Format plot
        ax.set_yticks(range(1, len(self.players_data) + 1))
        ax.set_yticklabels([p for p, _ in sorted(self.players_data.items())])
        ax.set_xlabel('Time')
        ax.set_title('Position Data Availability Over Time')
        
        # Format x-axis to show time nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add a grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add data percentage to y-axis labels
        labels = ax.get_yticklabels()
        for i, label in enumerate(labels):
            player = label.get_text()
            if player in results_df['player'].values:
                percentage = results_df[results_df['player'] == player]['data_percentage'].values[0]
                label.set_text(f"{player} ({percentage:.1f}%)")
        
        ax.set_yticklabels(labels)
        
        # Add legend
        ax.plot([], [], 'o', color='green', label='Position data available')
        ax.legend(loc='upper right')
        
        # Add annotation explaining the binning
        bin_time_str = f"{bin_size_seconds:.1f} seconds"
        plt.annotate(f"Time bin size: {bin_time_str}", 
                     xy=(0.02, 0.98), 
                     xycoords='figure fraction',
                     backgroundcolor='black',
                     color='white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Print statistics
        print("\nPlayer Data Availability Statistics:")
        print(results_df[['player', 'id', 'data_percentage', 'bins_with_data', 'total_bins']])
        
        # Save if output file is specified
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
            
            # Save the statistics to CSV
            stats_file = output_file.replace('.png', '_stats.csv')
            results_df.to_csv(stats_file, index=False)
            print(f"Saved statistics to {stats_file}")
        
        return fig, ax, results_df
    
    def create_data_gaps_report(self, output_file):
        """
        Create a detailed report analyzing gaps in player position data.
        
        Args:
            output_file (str): Path to save the output report
        """
        # Load data if needed
        self.load_data_if_needed()
        
        report_lines = ["=== PLAYER DATA GAPS ANALYSIS ===\n"]
        report_lines.append(f"Global time range: {self.global_min_time} to {self.global_max_time}")
        report_lines.append(f"Total duration: {(self.global_max_time - self.global_min_time).total_seconds():.1f} seconds\n")
        
        for player_name, player_info in sorted(self.players_data.items()):
            try:
                # Get player dataframe and ID
                if isinstance(player_info, dict) and 'data' in player_info:
                    df = player_info['data']
                    player_id = player_info.get('id', player_name.split()[-1])
                elif isinstance(player_info, pd.DataFrame):
                    df = player_info
                    player_id = player_name.split()[-1] if len(player_name.split()) > 1 else "Unknown"
                else:
                    continue
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Calculate time differences between consecutive points
                df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
                
                # Calculate position data presence
                df['has_position'] = df['x'].notna() & df['y'].notna()
                
                # Find long gaps (more than 5 seconds)
                long_gaps = df[df['time_diff'] > 5].copy()
                
                # Find periods with missing position data
                position_gaps = df[~df['has_position']].copy()
                
                report_lines.append(f"\n== {player_name} (ID: {player_id}) ==")
                report_lines.append(f"Total data points: {len(df)}")
                report_lines.append(f"First timestamp: {df['timestamp'].min()}")
                report_lines.append(f"Last timestamp: {df['timestamp'].max()}")
                report_lines.append(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.1f} seconds")
                
                # Calculate coverage relative to global time range
                player_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                global_duration = (self.global_max_time - self.global_min_time).total_seconds()
                coverage_percentage = (player_duration / global_duration) * 100 if global_duration > 0 else 0
                report_lines.append(f"Coverage of global time range: {coverage_percentage:.1f}%")
                
                # Report on time gaps
                if not long_gaps.empty:
                    report_lines.append(f"\nFound {len(long_gaps)} time gaps > 5 seconds:")
                    for idx, row in long_gaps.head(10).iterrows():
                        prev_time = row['timestamp'] - timedelta(seconds=row['time_diff'])
                        report_lines.append(f"  Gap of {row['time_diff']:.1f} seconds between {prev_time} and {row['timestamp']}")
                    
                    if len(long_gaps) > 10:
                        report_lines.append(f"  ... and {len(long_gaps) - 10} more gaps")
                else:
                    report_lines.append("\nNo significant time gaps found.")
                
                # Report on position data gaps
                position_missing_count = (~df['has_position']).sum()
                position_missing_pct = (position_missing_count / len(df)) * 100
                
                report_lines.append(f"\nPosition data missing: {position_missing_count} points ({position_missing_pct:.1f}%)")
                
                if not position_gaps.empty and len(position_gaps) <= 10:
                    report_lines.append("Missing position timestamps:")
                    for idx, row in position_gaps.iterrows():
                        report_lines.append(f"  {row['timestamp']}")
                elif not position_gaps.empty:
                    report_lines.append(f"First 5 missing position timestamps (of {len(position_gaps)} total):")
                    for idx, row in position_gaps.head(5).iterrows():
                        report_lines.append(f"  {row['timestamp']}")
                
                report_lines.append("-" * 50)
                
            except Exception as e:
                report_lines.append(f"{player_name}: Error analyzing data: {str(e)}\n")
        
        # Write the report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved detailed gaps report to {output_file}")
        
    def run_full_analysis(self, output_prefix="player_analysis", bin_size_seconds=5):
        """
        Run a complete analysis of player data availability.
        
        Args:
            output_prefix (str): Prefix for output files
            bin_size_seconds (int): Size of time bins in seconds
            
        Returns:
            tuple: (fig, ax, results_df) - Figure, axes and results dataframe
        """
        # Create output paths
        visualization_file = f"{output_prefix}_availability.png"
        gaps_report_file = f"{output_prefix}_gaps_report.txt"
        
        # Run the analysis
        fig, ax, results_df = self.analyze_data_availability(
            bin_size_seconds=bin_size_seconds,
            output_file=visualization_file
        )
        
        # Create gaps report
        self.create_data_gaps_report(gaps_report_file)
        
        return fig, ax, results_df

# Main script to run the analysis
if __name__ == "__main__":
    # Set your data folder path
    data_folder = "/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/outputs/fetched_data"
    
    # Create the analyzer
    analyzer = PlayerDataAnalyzer(data_folder=data_folder)
    
    # Run the full analysis
    analyzer.run_full_analysis(output_prefix="player_analysis")
    
    print("Analysis complete!")