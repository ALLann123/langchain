#!/usr/bin/python3
import json
from google_apis import create_service

client_secret='client_secret.json'

def construct_google_calendar_client(client_secret):
    """
    Constructs a Google calendar API client
    Parameters:
        -client_secret(str): The path to the client secret JSON file

    Returns:
        -service:The Google Calendar API service instance.
    """
    API_NAME='calendar'
    API_VERSION='v3'
    SCOPES=['https://www.googleapis.com/auth/calendar']
    service=create_service(client_secret, API_NAME, API_VERSION, SCOPES)
    return service

calendar_service=construct_google_calendar_client(client_secret)

def create_calendar_list(calendar_name):
    """
    Create a new calendar list
    Parameters:
        calendar_name(str): The name of the new calendar list
    Returns:
        -dict: A dictionary containing the ID of the new calendar list
    """
    calendar_list={
        'summary':calendar_name
    }
    created_calendar_list=calendar_service.calendarlist().insert(body=calendar_list).execute()
    return created_calendar_list
