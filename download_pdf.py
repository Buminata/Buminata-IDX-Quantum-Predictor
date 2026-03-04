import requests

url = "https://www.idx.co.id/StaticData/NewsAndAnnouncement/ANNOUNCEMENTSTOCK/From_EREP/202603/74cc72f7e5_29d7ff8853.pdf"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open("investor_ownership.pdf", "wb") as f:
            f.write(response.content)
        print("Download successful")
    else:
        print(f"Failed to download: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
