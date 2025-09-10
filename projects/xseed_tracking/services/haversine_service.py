from datetime import datetime
import math
import logging
from typing import Dict, List, Union, Optional
import numpy as np

class HaversineService:
    """A service class for calculating distances and related metrics between geographical coordinates."""
    
    def __init__(self):
        """Initialize the HaversineService."""
        self.distance = None
        

    @staticmethod
    def distance_haversine(p1: Dict[str, float], 
                    p2: Dict[str, float], 
                    meters: float = 1, 
                    ignore_status: bool = False,
                    rotation: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Calculate the Haversine distance between two points."""
        try:
            # Get the first value if it's a pandas Series, otherwise use the value directly
            lat1 = float(p1["lat"].iloc[0] if hasattr(p1["lat"], 'iloc') else p1["lat"])
            lng1 = float(p1["lng"].iloc[0] if hasattr(p1["lng"], 'iloc') else p1["lng"])
            lat2 = float(p2["lat"].iloc[0] if hasattr(p2["lat"], 'iloc') else p2["lat"])
            lng2 = float(p2["lng"].iloc[0] if hasattr(p2["lng"], 'iloc') else p2["lng"])

            earth_radius = 6378.1 * meters
            
            d_lat = math.radians(lat2 - lat1)
            d_lon = math.radians(lng2 - lng1)
            
            alpha = (lat2 - lat1) / 2
            beta = (lng2 - lng1) / 2
            
            a = (math.sin(math.radians(alpha)) ** 2 + 
                math.cos(math.radians(lat1)) * 
                math.cos(math.radians(lat2)) * 
                math.sin(math.radians(beta)) ** 2)
            c = math.asin(min(1, math.sqrt(a)))
            
            teta = math.atan2(
                math.sin(d_lon) * math.cos(math.radians(lat2)),
                math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
                math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                math.cos(d_lon)
            )
            
            distance = 2 * earth_radius * c
            
            # Use provided rotation or default to the hardcoded value
            rot_angle = -math.radians(rotation)
            dx = distance * math.cos(teta + rot_angle)
            dy = distance * math.sin(teta + rot_angle)
            
            return {
                "distance": round(distance, 3),
                "dx": dy,
                "dy": -dx,
                "teta": teta
            }
            
        except Exception as e:
            logging.warning(f"{__name__}: {str(e)}")
            return {
                "distance": None,
                "dx": None,
                "dy": None,
                "teta": None
            }

    @staticmethod
    def get_velocity(p1: Dict[str, str], 
                    p2: Dict[str, str], 
                    distance: float) -> Dict[str, float]:
        """
        Calculate velocity between two points given their timestamps and distance.
        
        Args:
            p1: Dict containing 'gps_time' key for first point
            p2: Dict containing 'gps_time' key for second point
            distance: Distance between points in meters
            
        Returns:
            Dict containing velocity and time difference in milliseconds
        """
        try:
            t1 = datetime.strptime(p1["gps_time"] + "0000", "%Y%m%d%H%M%S%f")
            t2 = datetime.strptime(p2["gps_time"] + "0000", "%Y%m%d%H%M%S%f")
            
            time_diff = (t2 - t1).total_seconds() * 1000  # milliseconds
            
            velocity = distance / (time_diff / 1000) if time_diff > 0 else 0.0
            
            return {
                "velocity": velocity,
                "diffInSeconds": time_diff
            }
        except Exception as e:
            logging.warning(f"{__name__}: {str(e)}")
            return {
                "velocity": 0.0,
                "diffInSeconds": 0.0
            }

    @staticmethod
    def get_pitch(p1: Dict[str, float], p2: Dict[str, float], rotation: Optional[float] = None) -> Dict[str, float]:
        """Calculate pitch metrics between two points using meters by default.
        
        Args:
            p1: Dict containing 'lat' and 'lng' keys for first point
            p2: Dict containing 'lat' and 'lng' keys for second point
            rotation: Optional rotation value in degrees
            
        Returns:
            Dict containing distance, dx, dy, and angle calculations in meters"""
        return HaversineService.distance_haversine(p1, p2, 1000, rotation=rotation)
    
    @staticmethod
    def matrix(asse_x: Dict[str, float], 
              asse_y: Dict[str, float], 
              coord: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Perform matrix transformation on coordinates.
        
        Args:
            asse_x: Dict containing normX and normY for X-axis transformation
            asse_y: Dict containing normX and normY for Y-axis transformation
            coord: Dict containing dx and dy coordinates
            
        Returns:
            Dict containing transformed x and y coordinates
        """
        if coord["dx"] is not None and coord["dy"] is not None:
            return {
                "x": (asse_x["normX"] * coord["dx"]) + (asse_x["normY"] * coord["dy"]),
                "y": (asse_y["normX"] * coord["dx"]) + (asse_y["normY"] * coord["dy"])
            }
        return {"x": None, "y": None}

    @staticmethod
    def polyfit(X: List[float], Y: List[float], n: int = 1) -> np.ndarray:
        """
        Fit an nth-order polynomial through a series of x-y data points using least squares.
        
        Args:
            X: List of x values
            Y: List of y values
            n: Order of polynomial to be used for fitting
            
        Returns:
            Array of polynomial coefficients
            
        Pre-Conditions: 
            The system is not underdetermined: len(X) > n+1
        """
        try:
            # Create the system of equations
            A = np.zeros((len(X), n + 1))
            for i in range(len(X)):
                for j in range(n + 1):
                    A[i, j] = X[i] ** j
                    
            # Create the B matrix
            B = np.array(Y).reshape(-1, 1)
            
            # Solve the system using numpy's least squares solver
            coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
            
            return coeffs.flatten()
            
        except Exception as e:
            logging.warning(f"Error in polyfit: {str(e)}")
            return np.array([])
