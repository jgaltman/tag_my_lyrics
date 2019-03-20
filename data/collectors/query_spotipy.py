import spotipy
username = 'joshlebed'
util.prompt_for_user_token(username,scope,client_id='your-app-redirect-url',client_secret='your-app-redirect-url',redirect_uri='your-app-redirect-url')
spotify = spotipy.Spotify()
name = 'kendrick lamar'
results = spotify.search(q='artist:' + name, type='artist')
print(results)