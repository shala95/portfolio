import json
import pandas as pd
from services.haversine_service import HaversineService
from services.pitch_service import PitchService
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def test_coordinate_transformation(json_file: str, stadium_file: str):
    """Compare PHP and Python coordinate transformations."""
    with open(json_file, 'r') as f:
        test_data = json.load(f)
    
    # Initialize services and get pitch data
    pitch_service = PitchService(stadium_file)
    pitch_result = pitch_service.get_pitch()
    
    # Get origin point
    origin = {
        "lat": pitch_result['origin']['lat'],
        "lng": pitch_result['origin']['lng']
    }
    
    differences = []
    for point in test_data:
        # PHP coordinates
        php_x = point['x']
        php_y = point['y']
        
        # Python transformation
        target_point = {"lat": point['lat'], "lng": point['lng']}
        result = HaversineService.distance_haversine(origin, target_point, meters=1000)
        
        if result['dx'] is not None and result['dy'] is not None:
            python_x = result['dx']
            python_y = result['dy']
            
            diff_x = abs(php_x - python_x)
            diff_y = abs(php_y - python_y)
            
            differences.append({
                'timestamp': point['gps_time'],
                'php_x': php_x,
                'php_y': php_y,
                'python_x': python_x,
                'python_y': python_y,
                'diff_x': diff_x,
                'diff_y': diff_y,
                'lat': point['lat'],
                'lng': point['lng']
            })
    
    df = pd.DataFrame(differences)
    
    stats = {
        'mean_diff_x': df['diff_x'].mean(),
        'mean_diff_y': df['diff_y'].mean(),
        'max_diff_x': df['diff_x'].max(),
        'max_diff_y': df['diff_y'].max(),
        'rmse_x': np.sqrt((df['diff_x']**2).mean()),
        'rmse_y': np.sqrt((df['diff_y']**2).mean())
    }
    
    return df, stats

def plot_coordinate_comparison(csv_file: str):
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PHP coordinates
    ax1.scatter(df['php_x'], df['php_y'], c='blue', label='PHP Implementation')
    ax1.set_title('PHP Implementation')
    ax1.set_xlabel('X coordinate (m)')
    ax1.set_ylabel('Y coordinate (m)')
    ax1.grid(True)
    
    # Python coordinates
    ax2.scatter(df['python_x'], df['python_y'], c='red', label='Python Implementation')
    ax2.set_title('Python Implementation')
    ax2.set_xlabel('X coordinate (m)')
    ax2.set_ylabel('Y coordinate (m)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('coordinate_comparison.png')
    plt.show()

if __name__ == "__main__":
    df, stats = test_coordinate_transformation(
        '/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/data/split_1.json',
        '/Users/mohamedshoala/Documents/internship/GPS/xseed_tracking/data/stadium_properties.csv'
    )
    
    print("\nCoordinate Transformation Comparison:")
    print(f"Mean X difference: {stats['mean_diff_x']:.6f} meters")
    print(f"Mean Y difference: {stats['mean_diff_y']:.6f} meters")
    print(f"Max X difference: {stats['max_diff_x']:.6f} meters")
    print(f"Max Y difference: {stats['max_diff_y']:.6f} meters")
    print(f"RMSE X: {stats['rmse_x']:.6f} meters")
    print(f"RMSE Y: {stats['rmse_y']:.6f} meters")
    
    df.to_csv('coordinate_comparison.csv', index=False)

    plot_coordinate_comparison('coordinate_comparison.csv')