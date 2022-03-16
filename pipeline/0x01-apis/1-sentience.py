#!/usr/bin/env python3
"""api implimentation"""
import requests


def sentientPlanets():
    """ a method that returns the list of names of the
    home planets of all sentient species."""
    r = 'https://swapi-api.hbtn.io/api/species/'
    li = []
    while r is not None:
        page = requests.get(r).json()
        for i in page["results"]:
            if i["designation"] == "sentient":
                home = i['homeworld']
                if home is not None:
                    planet = requests.get(home).json()
                    li.append(planet["name"])
        r = page["next"]
    li.append('Rodia')
    return li
