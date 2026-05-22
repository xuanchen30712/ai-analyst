import os
from webexteamssdk import WebexTeamsAPI

token = os.getenv("WEBEX_TOKEN")
api = WebexTeamsAPI(access_token=token)

search_term = "Cisco IQ"  # Change this to filter by different pattern

print(f"Rooms matching '{search_term}':")
for r in api.rooms.list():
    if search_term.lower() in (r.title or "").lower():
        print(f"  Title: {r.title}")
        print(f"  ID:    {r.id}")
        print()