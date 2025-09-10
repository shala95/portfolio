import pandas as pd

class CoordinateCleaner:
    @staticmethod
    def clean_coordinates(df):
        # Preview original data (first few rows)
        print("Original latitude and longitude sample:")
        print(df[['latitude', 'longitude']].head())

        # Use regex to keep only the first dot in latitude and longitude
        df['latitude'] = df['latitude'].str.replace(r'(?<=\..*)\.', '', regex=True)
        df['longitude'] = df['longitude'].str.replace(r'(?<=\..*)\.', '', regex=True)

        # Convert to numeric after reformatting
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # Validate conversion by checking for NaN values created due to errors
        lat_nan_count = df['latitude'].isna().sum()
        lng_nan_count = df['longitude'].isna().sum()
        print(f"Converted {lat_nan_count} latitude values and {lng_nan_count} longitude values to NaN due to non-numeric entries.")

        # Additional validation: check if values fall within valid latitude/longitude bounds
        invalid_lat = df[~df['latitude'].between(-90, 90)]
        invalid_lng = df[~df['longitude'].between(-180, 180)]
        if not invalid_lat.empty or not invalid_lng.empty:
            print("Warning: Found out-of-bounds values in latitude or longitude after conversion:")
            print("Invalid latitude values:")
            print(invalid_lat[['latitude']])
            print("Invalid longitude values:")
            print(invalid_lng[['longitude']])

        # Drop rows with NaN values in latitude or longitude
        df.dropna(subset=['latitude', 'longitude'], inplace=True)

        return df
