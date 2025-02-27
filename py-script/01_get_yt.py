# Packages
from pytubefix import YouTube
from pytubefix import Playlist
from pytubefix.cli import on_progress
from urllib.error import HTTPError
import os 

# Defining directory
interview_dir = "/work/pytube test/data/audio_output/interviews_from_JR"

# Define playlist from Jonas Risvigs collected interviews
playlist_interviews_JR = Playlist("https://www.youtube.com/watch?v=78MrFjlfo6w&list=PL2XPXGyI_pGOuilA9TpaPIOB34ziju7c_")
print('Number of videos in playlist: %s' % len(playlist_interviews_JR.video_urls))

for video in playlist_interviews_JR.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=interview_dir) # donwloading

# Downloading masterclass playlist from Jonas Risvig
masterclass_dir = "/work/pytube test/data/audio_output/interviews_from_JR/masterclass"

masterclass = Playlist("https://www.youtube.com/watch?v=vU5Hd7wBxes&list=PL2XPXGyI_pGNyi_4CjbnEeMq1fDwAtKBP")

for video in masterclass.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=masterclass_dir)

# Downloading sole video missing from the playlists
download_audio = "/work/pytube test/data/audio_output"

# Youtube object
yt = YouTube("https://www.youtube.com/watch?v=4mHW0Ff73Eg&ab_channel=P3xDRTV",
on_progress_callback=on_progress)

stream = yt.streams.filter(only_audio=True).first()

stream.download(output_path=download_audio)

# Behind the scences for kontra
kontra_dir = '/work/Ccp-MePSDA/data/audio_output/kontra'
os.makedirs(kontra_dir, exist_ok=True)

kontra = Playlist('https://www.youtube.com/playlist?list=PL2XPXGyI_pGNCuTee09chGu-nBwbaVxHK')

for video in kontra.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=kontra_dir) # donwloading

# Behind the scences for Evigt
evigt_dir = '/work/Ccp-MePSDA/data/audio_output/evigt' #define dir
os.makedirs(evigt_dir, exist_ok=True) # Creating playlist

evigt = Playlist('https://www.youtube.com/playlist?list=PL2XPXGyI_pGMKN2dX5loap5pA8PuiIE3S')

for video in evigt.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=evigt_dir) #download audio

# Behind the scenes for Zusa
zusa_dir = '/work/Ccp-MePSDA/data/audio_output/zusa'
os.makedirs(zusa_dir, exist_ok=True)

zusa = Playlist('https://www.youtube.com/watch?v=DC5mhNOQpTU&list=PL2XPXGyI_pGO2cfJ-VSuPdok-eef7f99o')

for video in zusa.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=zusa_dir) #download audio

# Behind the scenes Grænser
grænser_dir = '/work/Ccp-MePSDA/data/audio_output/grænser'
os.makedirs(grænser_dir, exist_ok=True)
# Link object
grænser = Playlist('https://www.youtube.com/watch?v=XHHUQqBl7ls&list=PL2XPXGyI_pGN-ADhYwtjohCA2YaDPKEcT')

for video in grænser.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=grænser_dir) #download audio

# Behind the scenes Grænser III
grænser_III_dir = '/work/Ccp-MePSDA/data/audio_output/grænser_III'
os.makedirs(grænser_III_dir, exist_ok=True)
# Link object
grænser_III = Playlist('https://www.youtube.com/playlist?list=PL2XPXGyI_pGN5HAPh_f8glS8Lb-HRKyEJ')

for video in grænser_III.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=grænser_III_dir) #download audio