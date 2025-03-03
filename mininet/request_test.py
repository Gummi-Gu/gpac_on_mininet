import requests
try:
    response = requests.get('http://127.0.0.1:12567/file/dash_tiled.mpd')
    print(response.text)
except requests.exceptions.Timeout as e:
    print('Timeout error:', e)
except requests.exceptions.TooManyRedirects as e:
    print('Too many redirects:', e)
except requests.exceptions.RequestException as e:
    print('Request error:', e)
except Exception as e:
    print('Unknown error:', e)