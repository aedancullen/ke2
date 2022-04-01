import string
import csv
import requests
import subprocess

url_prefix = "/watch?v="
session = requests.Session()

def process(name):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"}
    url = "https://www.youtube.com/results"
    response = session.get(url, params={"search_query": name + " audio"}, headers=headers).text
    idx = response.find(url_prefix)
    video_id = response[idx + len(url_prefix) : idx + len(url_prefix) + 11]
    topurl = "https://www.youtube.com" + url_prefix + video_id
    print("====>", name, topurl)
    subprocess.run('yt-dlp -f bestaudio[asr=44100] --extract-audio --audio-format mp3 --output "dl/' + name + '.%(ext)s" ' + topurl, shell=True, stdout=subprocess.DEVNULL)

names = []

with open("hot100.csv", 'r') as file:
    csv = csv.reader(file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    next(csv) # clear first line
    for line in csv:
        url,weekid,week_position,song,performer,songid,instance,previous_week_position,peak_position,weeks_on_chart = tuple(line)
        year = int(weekid[-4:])
        song = ''.join([c for c in song if c in string.ascii_letters or c in string.whitespace or c in string.digits])
        performer = ''.join([c for c in performer if c in string.ascii_letters or c in string.whitespace or c in string.digits])
        name = song + ' ' + performer
        if not year >= 2000:
            continue
        if not name in names:
            names.append(name)
            process(name)
