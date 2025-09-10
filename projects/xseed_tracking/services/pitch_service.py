from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
import logging
from services.haversine_service import HaversineService

@dataclass
class Stadium:
    """Data class for stadium coordinates and properties."""
    lat_p1: float
    lng_p1: float
    lat_p2: float
    lng_p2: float
    lat_p3: float
    lng_p3: float
    lat_p4: float
    lng_p4: float
    indoor: int = 0
    stadium_id: Optional[int] = None
    name: Optional[str] = None
    rotation: Optional[float] = None
    lato_lungo: Optional[float] = None
    lato_corto: Optional[float] = None
    pitch_width: Optional[float] = None
    pitch_height: Optional[float] = None

class PitchService:
    """Service for processing stadium pitch data and coordinates."""
    
    def __init__(self, stadium_path: Optional[str] = None):
        """Initialize PitchService."""
        self.points: List[Dict] = []
        self.data: List[Dict] = []
        self.f_pitch_scale = 1.1
        self.origin = None
        self.asseX = None
        self.asseY = None
        self.pitch_width = None
        self.pitch_height = None
        self.indoor = 0
        self.stadium = None
        self.rotate_pitch = False
        self.coords_map = {}
        self.rotation = None
        
        if stadium_path:
            self.set(stadium_path)

    def set(self, stadium_path: str) -> None:
        """
        Load and process stadium data from CSV.
        Handles both POINT format coordinates and additional stadium properties.
        """
        try:
            # Load stadium data
            stadium_data = pd.read_csv(stadium_path)
            row = stadium_data.iloc[0]
            
            def extract_coords(point_str: str) -> Dict[str, float]:
                """Extract coordinates from POINT format string."""
                # Remove 'POINT' and extract coordinates
                coords_str = point_str.replace('POINT', '').strip('() ').split()
                return {
                    "lat": float(coords_str[0]),
                    "lng": float(coords_str[1])
                }
            
            # Extract coordinates from POINT strings
            p1 = extract_coords(row['p1'])
            p2 = extract_coords(row['p2'])
            p3 = extract_coords(row['p3'])
            p4 = extract_coords(row['p4'])
            
            # Handle missing or NaN values with defaults
            indoor = 0  # Default value
            if 'indoor' in row and not pd.isna(row['indoor']):
                indoor = int(row['indoor'])
                
            stadium_id = None
            if 'id' in row and not pd.isna(row['id']):
                stadium_id = int(row['id'])
                
            name = row.get('stadium_name', None)
            if pd.isna(name):
                name = None
                
            # Store rotation value from stadium properties
            if 'rotation' in row and not pd.isna(row['rotation']):
                self.rotation = float(row['rotation'])
            else:
                self.rotation = -355.5093079  # Default if not in file
                
            lato_lungo = None
            if 'LatoLungo' in row and not pd.isna(row['LatoLungo']):
                lato_lungo = float(row['LatoLungo'])
                
            lato_corto = None
            if 'LatoCorto' in row and not pd.isna(row['LatoCorto']):
                lato_corto = float(row['LatoCorto'])
                
            pitch_width = None
            if 'pitch_width' in row and not pd.isna(row['pitch_width']):
                pitch_width = float(row['pitch_width'])
                
            pitch_height = None
            if 'pitch_height' in row and not pd.isna(row['pitch_height']):
                pitch_height = float(row['pitch_height'])
            
            # Create Stadium object with all available properties
            self.stadium = Stadium(
                lat_p1=p1["lat"],
                lng_p1=p1["lng"],
                lat_p2=p2["lat"],
                lng_p2=p2["lng"],
                lat_p3=p3["lat"],
                lng_p3=p3["lng"],
                lat_p4=p4["lat"],
                lng_p4=p4["lng"],
                indoor=indoor,
                stadium_id=stadium_id,
                name=name,
                rotation=self.rotation,  # Use the stored rotation value
                lato_lungo=lato_lungo,
                lato_corto=lato_corto,
                pitch_width=pitch_width,
                pitch_height=pitch_height
            )
            
            self.indoor = self.stadium.indoor
            
            # Initialize points array
            self.set_points()
            
            # Calculate pitch data
            data = []
            for point_pair in self.points:
                data.append(HaversineService.get_pitch(point_pair["p1"], point_pair["p2"], rotation=self.rotation))
            
            # Insert null point for calculations
            data.insert(1, {
                "distance": None,
                "dx": 0.0,
                "dy": 0.0,
                "teta": None
            })
            
            self.data = data
            
            # Set pitch dimensions if available from file
            if self.stadium.pitch_width and self.stadium.pitch_height:
                self.pitch_width = self.stadium.pitch_width
                self.pitch_height = self.stadium.pitch_height
            
        except Exception as e:
            logging.error(f"Error setting stadium data: {str(e)}")
            logging.error(f"Failed to process row: {row if 'row' in locals() else 'No row data'}")  # Debug info
            raise ValueError(f"Error processing stadium data: {str(e)}")

    def set_points(self) -> None:
        """Initialize points array and handle pitch rotation if needed."""
        points = [
            {
                "p1": {
                    "lat": self.stadium.lat_p2,
                    "lng": self.stadium.lng_p2
                },
                "p2": {
                    "lat": self.stadium.lat_p3,
                    "lng": self.stadium.lng_p3
                }
            },
            {
                "p1": {
                    "lat": self.stadium.lat_p2,
                    "lng": self.stadium.lng_p2
                },
                "p2": {
                    "lat": self.stadium.lat_p1,
                    "lng": self.stadium.lng_p1
                }
            }
        ]

        # Calculate dimensions
        tmp_lato_corto = HaversineService.get_pitch(points[0]["p1"], points[0]["p2"], rotation=self.rotation)["distance"]
        tmp_lato_lungo = HaversineService.get_pitch(points[1]["p1"], points[1]["p2"], rotation=self.rotation)["distance"]

        # Rotate pitch if needed
        if tmp_lato_corto > tmp_lato_lungo:
            self.rotate_pitch = True
            points = [
                {
                    "p1": {
                        "lat": self.stadium.lat_p3,
                        "lng": self.stadium.lng_p3
                    },
                    "p2": {
                        "lat": self.stadium.lat_p4,
                        "lng": self.stadium.lng_p4
                    }
                },
                {
                    "p1": {
                        "lat": self.stadium.lat_p3,
                        "lng": self.stadium.lng_p3
                    },
                    "p2": {
                        "lat": self.stadium.lat_p2,
                        "lng": self.stadium.lng_p2
                    }
                }
            ]

        self.points = points

    def set_coords_map(self, coords: List[Dict]) -> None:
        """Map coordinates to pitch reference points."""
        self.coords_map = {
            "PR": coords[3],
            "P1": coords[2],
            "P2": coords[1],
            "P3": coords[0]
        }

    def get_vertexes_coords(self, data: List[Dict]) -> Dict:
        """Calculate vertex coordinates of the pitch."""
        try:
            # Convert data to numpy array for calculations
            m = np.array([[d['dx'], d['dy']] for d in data if None not in (d['dx'], d['dy'])])
            if len(m) < 2:
                return False
                
            # Calculate slopes and intercepts
            m1 = np.polyfit(m[:, 1], m[:, 0], 1)[0]
            m2 = m1
            q2 = data[2]["dx"] - (m2 * data[2]["dy"])
            
            m3 = -1 / m1
            q3 = 0
            
            # Calculate intersection point
            x1 = (-q3 + q2) / (m3 - m2)
            x2 = q2 + (m2 * x1)
            
            # Transform coordinates
            coords = []
            for coord in data:
                coords.append({
                    "dx": coord["dy"],
                    "dy": coord["dx"]
                })
            
            coords.append({
                "dx": x1,
                "dy": x2
            })
            
            return {
                "coords": coords,
                "line_data": {
                    "m1": m1,
                    "m2": m2,
                    "m3": m3,
                    "q3": q3,
                    "q2": q2
                }
            }
        except Exception as e:
            logging.error(f"Error calculating vertices: {str(e)}")
            return False

    def get_xy_axis(self) -> Dict:
        """Calculate pitch axes for coordinate transformation."""
        asseX = {
            "x": self.coords_map["P3"]["dx"] - self.coords_map["P2"]["dx"],
            "y": self.coords_map["P3"]["dy"] - self.coords_map["P2"]["dy"]
        }

        asseY = {
            "x": self.coords_map["PR"]["dx"] - self.coords_map["P2"]["dx"],
            "y": self.coords_map["PR"]["dy"] - self.coords_map["P2"]["dy"]
        }

        normX = math.sqrt(pow(asseX["x"], 2) + pow(asseX["y"], 2))
        normY = math.sqrt(pow(asseY["x"], 2) + pow(asseY["y"], 2))

        asseX["normX"] = asseX["x"] / normX
        asseX["normY"] = asseX["y"] / normX
        
        asseY["normX"] = asseY["x"] / normY
        asseY["normY"] = asseY["y"] / normY

        self.asseX = asseX
        self.asseY = asseY

        return {
            "asseX": asseX,
            "asseY": asseY,
            "normY": asseY
        }

    def get_relative_pitch(self, asseX: Dict, asseY: Dict) -> Dict:
        """Calculate relative pitch coordinates."""
        return {
            "p3R": HaversineService.matrix(asseX, asseY, self.coords_map["P3"]),
            "p2R": HaversineService.matrix(asseX, asseY, self.coords_map["P2"]),
            "p1R": HaversineService.matrix(asseX, asseY, self.coords_map["P1"]),
            "p4R": HaversineService.matrix(asseX, asseY, self.coords_map["PR"])
        }

    def extend_relative_pitch(self, pR: Dict) -> Dict:
        """Scale pitch coordinates by pitch scale factor."""
        return {
            key: {
                "x": pR[orig_key]["x"] * self.f_pitch_scale,
                "y": pR[orig_key]["y"] * self.f_pitch_scale
            }
            for key, orig_key in zip(
                ["p1Rbig", "p2Rbig", "p3Rbig", "p4Rbig"],
                ["p1R", "p2R", "p3R", "p4R"]
            )
        }

    def set_pitch_dimensions(self, data: List[Dict]) -> None:
        """Set pitch dimensions and origin point."""
        # Use provided dimensions if available, otherwise calculate
        if not (self.pitch_width and self.pitch_height):
            self.pitch_height = data[2]["distance"]
            self.pitch_width = data[0]["distance"]
        
        self.origin = {
            "lat": self.points[0]["p1"]["lat"],
            "lng": self.points[0]["p1"]["lng"]
        }

    def get_side_lengths(self, pR: Dict, pRbig: Dict) -> Dict:
        """Calculate pitch side lengths."""
        return {
            "LatoCorto": math.ceil(pR["p3R"]["x"]),
            "LatoLungo": math.ceil(pR["p4R"]["y"]),
            "LatoCortoBig": math.ceil(pRbig["p3Rbig"]["x"]),
            "LatoLungoBig": math.ceil(pRbig["p4Rbig"]["y"])
        }

    def center_pitch(self, pitch: Dict, shift: Dict) -> Dict:
        """Center the pitch using calculated shift."""
        return {
            f"p{i}Rbig": {
                "x": pitch[f"p{i}Rbig"]["x"] - shift["x"],
                "y": pitch[f"p{i}Rbig"]["y"] - shift["y"]
            }
            for i in range(1, 5)
        }

    @staticmethod
    def media(a: float, b: float, c: float, d: float) -> float:
        """Calculate average of four values."""
        return (a + b + c + d) / 4
    def get_pitch(self) -> Dict:
        """Calculate and return complete pitch data."""
        self.set_pitch_dimensions(self.data)
        vertexes = self.get_vertexes_coords(self.data)
        
        if not vertexes:
            raise ValueError("Could not calculate vertex coordinates")
            
        coords = vertexes['coords']
        self.set_coords_map(coords)
        
        axes = self.get_xy_axis()
        pR = self.get_relative_pitch(axes["asseX"], axes["asseY"])
        pRbig = self.extend_relative_pitch(pR)
        side_lengths = self.get_side_lengths(pR, pRbig)

        # Calculate shift for centering
        shift = {
            "x": self.media(*(pRbig[f"p{i}Rbig"]["x"] for i in range(1, 5))) -
                 self.media(*(pR[f"p{i}R"]["x"] for i in range(1, 5))),
            "y": self.media(*(pRbig[f"p{i}Rbig"]["y"] for i in range(1, 5))) -
                 self.media(*(pR[f"p{i}R"]["y"] for i in range(1, 5)))
        }
        
        pRbig = self.center_pitch(pRbig, shift)

        return {
            "LatoCorto": side_lengths["LatoCorto"],
            "LatoLungo": side_lengths["LatoLungo"],
            "LatoLungoBig": side_lengths["LatoLungoBig"],
            "LatoCortoBig": side_lengths["LatoCortoBig"],
            "pR": pR,
            "pRbig": pRbig,
            "origin": self.origin,
            "second_point": self.points[0]["p2"],
            "normX": axes["asseX"]["x"],
            "normY": axes["normY"],
            "pitch_width": self.pitch_width,
            "pitch_height": self.pitch_height,
            "asseX": axes["asseX"],
            "asseY": axes["asseY"],
            "points": coords,
            "m1": vertexes["line_data"]["m1"],
            "q2": vertexes["line_data"]["q2"],
            "m3": vertexes["line_data"]["m3"],
            "q3": vertexes["line_data"]["q3"],
            "teta": self.data[0]["teta"],
            "L1": self.data[0]["distance"],
            "L2": self.data[2]["distance"],
            "stadium_info": {
                "id": self.stadium.stadium_id,
                "name": self.stadium.name,
                "rotation": self.stadium.rotation,
                "indoor": self.stadium.indoor
            }
        }

    def get_indoor(self) -> int:
        """Get indoor status."""
        return self.indoor

    def set_pitch_scale(self, f_pitch_scale: float) -> None:
        """Set pitch scale factor."""
        self.f_pitch_scale = f_pitch_scale

    def get_origin(self) -> Dict:
        """Get origin point coordinates."""
        return self.origin

    def get_asseXY(self) -> Dict:
        """Get pitch axes data."""
        return {
            "asseX": self.asseX,
            "asseY": self.asseY
        }