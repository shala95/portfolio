from src.data_processor import DataProcessor
from src.display_manager import DisplayManager
from src.gps_comparison import GPSComparison
from src.pitch_service import PitchService

def main():
    # Initialize comparison object
    data_proc       = DataProcessor(stadium_path = 'data/stadium_properties.csv')
    gps_comparison  = GPSComparison(stadium_path = 'data/stadium_properties.csv')
    results = gps_comparison.analyze_all_players()

if __name__ == "__main__":
    main()