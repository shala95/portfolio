import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
from math import sin, cos, asin, sqrt, atan2, degrees, radians
from haversine import haversine_vector, Unit

from config_manager import ConfigManager
config = ConfigManager('config.yaml').load_config()

@dataclass
class EarthConstants:
    RADIUS: float = 6371000  # Earth's mean radius in meters

class EnhancedCoordinateProcessor:
    def __init__(self, lat_p2: float, lng_p2: float, lat_p3: float, lng_p3: float):
        """Initialize processor with reference points."""
        self.lat_p2 = lat_p2
        self.lng_p2 = lng_p2
        self.lat_p3 = lat_p3
        self.lng_p3 = lng_p3
        
        self.lat_p2_rad = math.radians(lat_p2)
        self.lng_p2_rad = math.radians(lng_p2)
        self.lat_p3_rad = math.radians(lat_p3)
        self.lng_p3_rad = math.radians(lng_p3)
        
        self.earth = EarthConstants()
        self.rotation_angle_wrt_north = self.calculate_bearing_from_latlong((self.lat_p2, self.lng_p2), (self.lat_p3, self.lng_p3))


    def calculate_bearing_from_latlong(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate the bearing between two points.
        
        Parameters
        ----------
        point1 : Tuple[float, float]
            Latitude and longitude of the first point.
        point2 : Tuple[float, float]
            Latitude and longitude of the second point.
        
        Returns
        -------
        float
            bearing in degrees from the first point to the second point, with respect to the north (clockwise from the north)
        """
        lat1, lon1 = map(radians, point1)
        lat2, lon2 = map(radians, point2)
        
        delta_lon = lon2 - lon1
        
        x = sin(delta_lon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(delta_lon))
        
        initial_bearing = atan2(x, y)
        
        # Convert from radians to degrees and normalize to config['ahrs_filters']['angle_range']
        initial_bearing = degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) 

        #_normalize_angle
        compass_bearing  = self._normalize_angle(compass_bearing)
        
        return compass_bearing


    # def _calculate_bearing(self) -> float:
    #     """Calculate bearing angle between P2 and P3."""
    #     d_lon = self.lng_p3_rad - self.lng_p2_rad
        
    #     y = math.sin(d_lon) * math.cos(self.lat_p3_rad)
    #     x = (math.cos(self.lat_p2_rad) * math.sin(self.lat_p3_rad) - 
    #          math.sin(self.lat_p2_rad) * math.cos(self.lat_p3_rad) * 
    #          math.cos(d_lon))
        
    #     return math.atan2(y, x)

    @staticmethod
    def _normalize_angle(angle: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """
        Normalize angles based on the configuration.

        Args:
            angle: Angle or array-like of angles to normalize.

        Returns:
            Normalized angle(s) in the range specified by config.

        Raises:
            ValueError: If the angle range in config is not '360' or '180'.
        """
        angle_range = config['ahrs']['angle_range']
        
        if angle_range not in ['360', '180']:
            raise ValueError(f"Unknown angle range: {angle_range}")
        
        normalized = np.mod(angle, 360)
        
        if angle_range == '180':
            normalized = np.where(normalized > 180, normalized - 360, normalized)
        
        return normalized
    
    def transform_to_local_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        """Transform lat/lon to local X/Y coordinates."""
        try:
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            
            d_lat = lat_rad - self.lat_p2_rad
            d_lon = lon_rad - self.lng_p2_rad
            
            mean_lat = (lat_rad + self.lat_p2_rad) / 2
            
            y = d_lat * self.earth.RADIUS
            x = d_lon * self.earth.RADIUS * math.cos(mean_lat)
            
            return x, y
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return 0.0, 0.0

    def transform_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform all coordinates in DataFrame."""
        transformed_data = []
        
        for _, row in df.iterrows():
            x, y = self.transform_to_local_coordinates(
                row['latitude'],
                row['longitude']
            )
            transformed_data.append({
                'date time': row['date time'],
                'x_real_m': x,
                'y_real_m': y,
                'latitude': row['latitude'],
                'longitude': row['longitude']
            })
        
        return pd.DataFrame(transformed_data)

    def process_and_save(self, input_data: str, output_data: str) -> None:
        """Load, transform, and save data."""
        try:
            df = pd.read_csv(input_data, sep=";")
            df_transformed = self.transform_coordinates(df)
            df_transformed.to_csv(output_data, index=False)
            print(f'Transformed file saved to {output_data}')
        except Exception as e:
            print(f"Error processing file: {e}")
            raise

    def convert_pitch_corners(self, pitch_file: str) -> Optional[pd.DataFrame]:
        """Convert pitch corner coordinates."""
        try:
            pitch_df = pd.read_excel(pitch_file, engine='openpyxl')
            return self.transform_coordinates(pitch_df)
        except Exception as e:
            print(f"Error converting pitch corners: {e}")
            return None

    def validate_with_google_maps_file(self, validation_file: str) -> Dict[str, float]:
        """Validate transformation using validation points."""
        try:
            df = pd.read_csv(validation_file)
            results = []
            
            for _, point in df.iterrows():
                calc_x, calc_y = self.transform_to_local_coordinates(
                    point['latitude'],
                    point['longitude']
                )
                
                error = math.sqrt(
                    (calc_x - point['google_maps_x'])**2 +
                    (calc_y - point['google_maps_y'])**2
                )
                
                results.append({
                    'description': point['description'],
                    'calculated_x': calc_x,
                    'calculated_y': calc_y,
                    'expected_x': point['google_maps_x'],
                    'expected_y': point['google_maps_y'],
                    'error': error
                })
            
            return {
                'mean_error': np.mean([r['error'] for r in results]),
                'max_error': max([r['error'] for r in results]),
                'results': results
            }
            
        except Exception as e:
            print(f"Validation error: {e}")
            raise

    def filter_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove coordinate outliers using z-score."""
        df_filtered = df.copy()
        
        for col in ['latitude', 'longitude']:
            z_scores = np.abs((df_filtered[col] - df_filtered[col].mean()) / df_filtered[col].std())
            df_filtered = df_filtered[z_scores < threshold]
        
        return df_filtered.reset_index(drop=True) 
    def calculate_path_length(self, df: pd.DataFrame) -> float:
        """Calculate total path length in meters."""
        total_length = 0
        for i in range(len(df)-1):
            x1, y1 = df.iloc[i]['x_real_m'], df.iloc[i]['y_real_m']
            x2, y2 = df.iloc[i+1]['x_real_m'], df.iloc[i+1]['y_real_m']
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_length += segment_length
        return total_length
    def validate_transformation(self) -> Dict[str, float]:
        """Validate the transformation using reference points."""
        x, y = self.transform_to_local_coordinates(self.lat_p3, self.lng_p3)
        total_distance = math.sqrt(x**2 + y**2)
        angle = math.degrees(math.atan2(y, x))
        return {
            'distance': total_distance,
            'angle': angle
        }