#!/usr/bin/env python3
"""api implimentation"""
import requests
import sys
import time


if __name__ == '__main__':
    resp = requests.get(sys.argv[1])
    if resp.status_code == 200:
        print(resp.json()["location"])
    elif resp.status_code == 403:
        x = int(time.time()) - int(resp.headers['X-Ratelimit-Reset'])
        print('Reset in {} min'.format(abs(x) / 60))
    else:
        print('Not found')
