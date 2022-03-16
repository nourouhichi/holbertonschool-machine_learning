#!/usr/bin/env python3
"""API implement spacex"""
import requests


if __name__ == '__main__':
    resp = requests.get('https://api.spacexdata.com/v4/launches')
    js = resp.json()
    dic = {}
    for i in js:
        rocket = i['rocket']
        rocket_url = "https://api.spacexdata.com/v4/rockets/" + rocket
        r_rocket = requests.get(rocket_url)
        r_rocket_get = r_rocket.json()
        rocket_name = r_rocket_get["name"]
        if rocket_name in dic.keys():
            dic[rocket_name] = dic[rocket_name] + 1
        else:
            dic[rocket_name] = 1
    for key, value in reversed(sorted(dic.items(),
                               key=lambda key: key[1])):
        print("{}: {}".format(key, value))
