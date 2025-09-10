from services.haversine_service import HaversineService
from services.pitch_service import PitchService

stadium_path = 'data/stadium_properties.csv'

# Load pitch parameters
pitch_service = PitchService(stadium_path)