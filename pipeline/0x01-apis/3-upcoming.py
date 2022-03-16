#!/usr/bin/env python3
"""API implement"""
import requests
import time


if __name__ == '__main__':
    resp = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    t = resp.json()[0]['date_unix']
    now = time.time()
    min = abs(t - now)
    upcom = resp.json()[0]
    for i in resp.json():
        if i['date_unix'] < min:
            min = i['date_unix']
            upcom = i
rocket = requests.get('https://api.spacexdata.com/v4/rockets/'
                      + i['rocket'])
lpad = requests.get('https://api.spacexdata.com/v4/launchpads/'
                    + i['launchpad'])
lpad = lpad.json()
locale = lpad['locality']
lpad = lpad['name']
print('{} ({}) {} - {} ({})'.format(i['name'], i['date_local'],
      rocket, lpad, locale))
