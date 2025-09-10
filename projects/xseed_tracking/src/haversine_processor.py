import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict

@dataclass
class EarthConstants:
    RADIUS: float = 6378.1  # Earth's radius in kilometers

class HaversineProcessor:
    def __init__(self, corner1_lat: float, corner1_lon: float, 
                 corner2_lat: float, corner2_lon: float):
        self.corner1_lat = corner1_lat
        self.corner1_lon = corner1_lon
        self.corner2_lat = corner2_lat
        self.corner2_lon = corner2_lon
        self.earth = EarthConstants()
        self.ref_angle = self.calculate_reference_angle()
        print(f"Reference angle: {math.degrees(self.ref_angle)} degrees")

    def set_pitch_result(self, pitch_result: Dict) -> None:
        """Set pitch result from PitchService."""
        self.pitch_result = pitch_result

    def haversine_distance_php_style(self, lat1: float, lon1: float, 
                                   lat2: float, lon2: float, 
                                   meters: int = 1000) -> Dict[str, float]:
        """Calculate distance and components using PHP-style Haversine formula."""
        earth_radius = self.earth.RADIUS * meters
        
        # Calculate differences
        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        
        # Convert differences to radians
        d_lat_rad = math.radians(d_lat)
        d_lon_rad = math.radians(d_lon)
        
        # Intermediate calculations as in PHP
        alpha = d_lat / 2
        beta = d_lon / 2
        
        # Haversine formula NOTE: Check if this is correct (np.sin vs math.sin)
        a = (math.sin(math.radians(alpha)) ** 2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(math.radians(beta)) ** 2)
        
        c = math.asin(min(1, math.sqrt(a)))
        
        # Calculate theta (bearing)
        theta = math.atan2(
            math.sin(d_lon_rad) * math.cos(math.radians(lat2)),
            math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.cos(d_lon_rad)
        )
        
        # Calculate distance and components
        distance = 2 * earth_radius * c
        dx = distance * math.sin(theta)
        dy = distance * math.cos(theta)
        theta_degrees = math.degrees(theta)
        
        return {
            "distance": round(distance, 3),
            "dx": dx,
            "dy": dy,
            "theta": theta
        }

    def matrix_transform(self, coord: Dict) -> Tuple[float, float]: #NOTE: What are asseX and asseY? Are they same as for FM? 
        """Apply matrix transformation to coordinates"""
        if not self.pitch_result or coord["dx"] is None or coord["dy"] is None:
            return 0.0, 0.0
            
        asse_x = self.pitch_result["asseX"]
        asse_y = self.pitch_result["asseY"]
        
        x = (asse_x["normX"] * coord["dx"]) + (asse_x["normY"] * coord["dy"])
        y = (asse_y["normX"] * coord["dx"]) + (asse_y["normY"] * coord["dy"])
        
        return x, y


    def calculate_reference_angle(self) -> float:
        """Calculate reference angle from corner coordinates."""
        result = self.haversine_distance_php_style(
            self.corner1_lat, self.corner1_lon,
            self.corner2_lat, self.corner2_lon
        )
        return result['theta']

    def transform_to_local_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert latitude/longitude to local coordinates using PHP-style transformation."""
        try:
            # PHP-style transformation
            point = {"lat": lat, "lng": lon}
            d = self.haversine_distance_php_style(
                self.corner1_lat, self.corner1_lon,
                lat, lon, 
                1000
            )
            
            return self.matrix_transform(d)
            
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return 0.0, 0.0

    def transform_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform coordinates while preserving all original columns."""
        df_transformed = df.copy()
        coordinates = [self.transform_to_local_coordinates(lat, lon) 
                      for lat, lon in zip(df['latitude'], df['longitude'])]
        df_transformed['x_real_m'] = [x for x, _ in coordinates]
        df_transformed['y_real_m'] = [y for _, y in coordinates]
        return df_transformed

    def standardize_timestamps(self, df: pd.DataFrame, date: str = "2024-10-29") -> pd.DataFrame:
        """Standardize timestamps to include date if missing."""
        df = df.copy()
        if 'date time' in df.columns:
            if isinstance(df['date time'].iloc[0], str) and len(df['date time'].iloc[0]) <= 12:
                df['date time'] = f"{date} " + df['date time']
            df['date time'] = pd.to_datetime(df['date time'])
        return df

    def process_and_save(self, input_data: str, output_data: str) -> None:
        """Load, transform, and save data to CSV."""
        try:
            df = pd.read_csv(input_data, sep=";")
            df = self.standardize_timestamps(df)
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
            pitch_df.rename(columns={'lat': 'latitude', 'lng': 'longitude'}, inplace=True)

            transformed_df = self.transform_coordinates(pitch_df)
            print("Pitch corners after transformation:")
            print(transformed_df[['x_real_m', 'y_real_m']])
            return transformed_df
        except Exception as e:
            print(f"Error converting pitch corners: {e}")
            return None

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
            total_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return total_length

    @staticmethod
    def calculate_distance(p1: Dict[str, float], p2: Dict[str, float]) -> Dict[str, float]:
        """Calculate distance and components using Haversine formula."""
        try:
            earth_radius = 6378.1 * 1000  # meters
            
            d_lat = p2["lat"] - p1["lat"]
            d_lon = p2["lng"] - p1["lng"]
            d_lon_rad = math.radians(d_lon)
            
            alpha = d_lat / 2
            beta = d_lon / 2
            
            a = (math.sin(math.radians(alpha)) * math.sin(math.radians(alpha)) + 
                 math.cos(math.radians(p1["lat"])) * math.cos(math.radians(p2["lat"])) * 
                 math.sin(math.radians(beta)) * math.sin(math.radians(beta)))
            
            c = math.asin(min(1, math.sqrt(a)))
            
            theta = math.atan2(
                math.sin(d_lon_rad) * math.cos(math.radians(p2["lat"])),
                math.cos(math.radians(p1["lat"])) * math.sin(math.radians(p2["lat"])) -
                math.sin(math.radians(p1["lat"])) * math.cos(math.radians(p2["lat"])) *
                math.cos(d_lon_rad)
            )
            
            distance = 2 * earth_radius * c
            dx = distance * math.sin(theta)
            dy = distance * math.cos(theta)
            
            return {
                "distance": round(distance, 3),
                "dx": dx,
                "dy": dy,
                "theta": theta
            }
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return {"distance": None, "dx": None, "dy": None, "theta": None}

    def validate_transformation(self) -> Dict[str, float]:
        """Validate the transformation using corner points."""
        p1 = {"lat": self.corner1_lat, "lng": self.corner1_lon}
        p2 = {"lat": self.corner2_lat, "lng": self.corner2_lon}
        
        result = self.calculate_distance(p1, p2)
        
        print("Validation results:")
        print(f"Distance between corners: {result['distance']:.2f}m")
        print(f"Rotation angle: {math.degrees(self.ref_angle):.2f} degrees")
        
        return {
            'distance': result['distance'],
            'angle': math.degrees(self.ref_angle),
            'dx': result['dx'],
            'dy': result['dy']
        }