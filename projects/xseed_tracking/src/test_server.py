import requests
import time

# API configuration
API_IP = "15.161.76.156"
API_HOSTNAME = "xseed-advanced-metrics-btb-env.eba-p2hm3mer.eu-south-1.elasticbeanstalk.com"
API_ENDPOINT = f"http://{API_IP}/trace"

# Test parameters (from one of your failing requests)
params = {
    "user_id": 11175,
    "session_id": 36815,
    "segment_id": 17916,
    "type": "drill",
    "version": 1
}

# Set headers
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Host': API_HOSTNAME  # Include original hostname for virtual hosting
}

print(f"Testing connection to {API_ENDPOINT}")
print(f"With parameters: {params}")
print("Sending request...")

try:
    # Try with a longer timeout
    response = requests.get(API_ENDPOINT, params=params, headers=headers, timeout=60)
    
    # Check status code
    print(f"Status code: {response.status_code}")
    
    # Print response content
    if response.status_code == 200:
        response_json = response.json()
        if response_json.get("data"):
            print(f"Success! Received data with {len(response_json['data'])} records")
        else:
            print("Request successful but no data returned")
            print(response_json)
    else:
        print(f"Error response: {response.text}")
        
except requests.exceptions.Timeout:
    print("The request timed out after 60 seconds")
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
    print("This suggests the server is unreachable or not accepting connections")
except Exception as e:
    print(f"Unexpected error: {e}")

# Try with a different type to compare
print("\n\nNow testing with 'match' type instead of 'drill'...")
params["type"] = "match"

try:
    # Try with a longer timeout
    response = requests.get(API_ENDPOINT, params=params, headers=headers, timeout=60)
    
    # Check status code
    print(f"Status code: {response.status_code}")
    
    # Print response content
    if response.status_code == 200:
        response_json = response.json()
        if response_json.get("data"):
            print(f"Success! Received data with {len(response_json['data'])} records")
        else:
            print("Request successful but no data returned")
            print(response_json)
    else:
        print(f"Error response: {response.text}")
        
except requests.exceptions.Timeout:
    print("The request timed out after 60 seconds")
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
    print("This suggests the server is unreachable or not accepting connections")
except Exception as e:
    print(f"Unexpected error: {e}")