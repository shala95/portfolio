from src.coordinates_processor import EnhancedCoordinateProcessor
from pathlib import Path

def run_validation():
    # Define reference points
    lat_p2 = 45.497414
    lng_p2 = 9.265013
    lat_p3 = 45.497440
    lng_p3 = 9.265458

    # Initialize processor
    processor = EnhancedCoordinateProcessor(lat_p2, lng_p2, lat_p3, lng_p3)

    print("\n=== Transformation Validation ===")
    processor.validate_transformation()

    print("\n=== Google Maps Points Validation ===")
    validation_file = './data/validation-points.csv'
    if Path(validation_file).exists():
        try:
            processor.validate_with_google_maps_file(validation_file)
        except Exception as e:
            print(f"Error during validation: {e}")
    else:
        print(f"Validation file not found at: {validation_file}")

if __name__ == "__main__":
    run_validation()
