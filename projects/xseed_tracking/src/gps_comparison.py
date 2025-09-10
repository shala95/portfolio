import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
from services.haversine_service import HaversineService
from services.pitch_service import PitchService

class GPSComparison:
    def __init__(self, stadium_path: str):
        
        # Load pitch parameters
        pitch_service = PitchService.from_csv(stadium_path, pitch_id=3200)
        self.pitch_result = pitch_service.get_pitch()

        # corner1_lat, corner1_lon = 45.49684878071292, 9.265076145529747  # top-left corner
        # corner2_lat, corner2_lon = 45.497406694764145, 9.26501378417015  # top-right corner
        corner1_lat, corner1_lon = self.pitch_result['origin']['lat'], self.pitch_result['origin']['lng']
        corner2_lat, corner2_lon = self.pitch_result['second_point']['lat'], self.pitch_result['second_point']['lng']

        self.processor = HaversineService(
            corner1_lat, corner1_lon,
            corner2_lat, corner2_lon
        )

        self.processor.set_pitch_result(self.pitch_result)  
        
        self.json_template = "./data/player_{}/shinguard_{}.json"
        self.csv_template = "./data/player_{}/processed_{}.csv"
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)

    def standardize_timestamp(self, timestamp_str: str) -> pd.Timestamp:
        if ',' in timestamp_str:
            timestamp_str = timestamp_str.split(',')[0]
        return pd.to_datetime(timestamp_str)

    def load_json_data(self, player_num: int, foot: str) -> pd.DataFrame:
        try:
            file_path = self.json_template.format(player_num, foot)
            
            # Use pandas to load the JSON data directly
            data = pd.read_json(file_path)
            
            # Preprocess and transform the data if needed
            # data['gps_time'] = data['gps_time'].apply(self.standardize_timestamp) #NOTE: why this? 
            data['gps_time'] = pd.to_datetime(data['gps_time'])

            data = data.rename(columns={
                'lat': 'latitude',
                'lng': 'longitude',
                'x': 'original_x',
                'y': 'original_y'
            })
            
            # Select only the required columns
            return data[['gps_time', 'latitude', 'longitude', 'original_x', 'original_y']]
            
        except Exception as e:
            print(f"Error loading JSON data for player {player_num}, {foot} foot: {e}")
            return pd.DataFrame()

    def load_csv_data(self, player_num: int, foot: str) -> pd.DataFrame:
        try:
            file_path = self.csv_template.format(player_num, foot)
            df = pd.read_csv(file_path)
            df['gps_time'] = pd.to_datetime(df['gps_time'])
            return df
            
        except Exception as e:
            print(f"Error loading CSV data for player {player_num}, {foot} foot: {e}")
            return pd.DataFrame()

    def process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'latitude' in df.columns and 'longitude' in df.columns:
            transformed = self.processor.transform_coordinates(df)
            return transformed
        return df

    def export_comparison_data(self, player_num: int, foot: str, 
                             transformed_df: pd.DataFrame, csv_df: pd.DataFrame) -> None:
        """Export transformed and comparison data to CSV files."""
        # Create player output directory
        player_dir = self.output_dir / f"player_{player_num}"
        player_dir.mkdir(exist_ok=True)

        # Prepare merged dataset
        merged_data = pd.merge_asof(
            transformed_df.sort_values('gps_time'),
            csv_df.sort_values('gps_time'),
            on='gps_time',
            tolerance=pd.Timedelta(milliseconds=100),
            suffixes=('_haversine', '_original')
        )

        # Add difference columns
        merged_data['x_difference'] = merged_data['x_real_m'] - merged_data['x']
        merged_data['y_difference'] = merged_data['y_real_m'] - merged_data['y']
        merged_data['total_difference'] = np.sqrt(
            merged_data['x_difference']**2 + merged_data['y_difference']**2
        )

        # Export files
        base_path = player_dir / f"{foot}_foot"
        merged_data.to_csv(f"{base_path}_comparison.csv", index=False)
        transformed_df.to_csv(f"{base_path}_haversine.csv", index=False)

    def compare_coordinates(self, transformed_df: pd.DataFrame, 
                          csv_df: pd.DataFrame, 
                          time_tolerance: pd.Timedelta = pd.Timedelta(milliseconds=100)) -> Dict[str, float]:
        differences = []
        matched_points = 0
        total_points = len(transformed_df)
        x_diffs = []
        y_diffs = []

        for _, our_point in transformed_df.iterrows():
            mask = (csv_df['gps_time'] - our_point['gps_time']).abs() <= time_tolerance
            matching_points = csv_df[mask]

            if not matching_points.empty:
                matched_point = matching_points.iloc[0]
                x_diff = our_point['x_real_m'] - matched_point['x']
                y_diff = our_point['y_real_m'] - matched_point['y']
                
                x_diffs.append(x_diff)
                y_diffs.append(y_diff)
                
                total_diff = np.sqrt(x_diff**2 + y_diff**2)
                differences.append(total_diff)
                matched_points += 1

        if differences:
            return {
                'mean_difference': float(np.mean(differences)),
                'max_difference': float(np.max(differences)),
                'rmse': float(np.sqrt(np.mean(np.array(differences)**2))),
                'std_difference': float(np.std(differences)),
                'x_bias': float(np.mean(x_diffs)),
                'y_bias': float(np.mean(y_diffs)),
                'matched_points': matched_points,
                'total_points': total_points,
                'matching_rate': (matched_points / total_points * 100)
            }
        return None

    def analyze_player_data(self, player_num: int, foot: str) -> Dict:
        json_data = self.load_json_data(player_num, foot)
        csv_data = self.load_csv_data(player_num, foot)
        
        if not json_data.empty and not csv_data.empty:
            transformed_data = self.process_coordinates(json_data)
            
            # Export the data for manual analysis
            self.export_comparison_data(player_num, foot, transformed_data, csv_data)
            
            results = {
                'haversine_vs_original': {
                    'x_correlation': float(np.corrcoef(transformed_data['x_real_m'], 
                                                     transformed_data['original_x'])[0,1]),
                    'y_correlation': float(np.corrcoef(transformed_data['y_real_m'], 
                                                     transformed_data['original_y'])[0,1])
                },
                'haversine_vs_processed': self.compare_coordinates(transformed_data, csv_data)
            }
            return results
        return None

    def analyze_all_players(self, player_numbers: List[int] = [1, 2, 3]) -> Dict:
        results = {}
        for player_num in player_numbers:
            player_results = {}
            for foot in ['left', 'right']:
                analysis_results = self.analyze_player_data(player_num, foot)
                player_results[foot] = analysis_results
            results[f'Player_{player_num}'] = player_results
        return results