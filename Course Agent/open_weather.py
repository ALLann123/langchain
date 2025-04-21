import requests
from dotenv import load_dotenv
import os

load_dotenv()

def get_weather(city):
    api_key=os.getenv("OPEN_WEATHER")
    #api_key = "667e86d4e60ee725acab5d3d05561d57"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url).json()
    
    print(f"Weather in {city}:")
    print(f"Temperature: {response['main']['temp']}°C")
    print(f"Condition: {response['weather'][0]['description']}")
    print(f"Humidity: {response['main']['humidity']}%")

# Call the function
get_weather("Nakuru")