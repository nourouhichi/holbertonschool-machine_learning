#!/usr/bin/env python3
"""api implimentation"""
import requests


def availableShips(passengerCount):
    """ a method that returns the list of ships that
    can hold a given number of passengers"""
    r = 'https://swapi-api.hbtn.io/api/starships/'
    li = []
    while r is not None:
        page = requests.get(r).json()
        for i in page["results"]:
            passen = i['passengers'].replace(',', '')
            if passen != 'n/a' and passen != 'unknown':
                if int(passen) >= passengerCount:
                    li.append(i["name"])
        r = page["next"]
    return li
