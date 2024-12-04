from pytubefix import YouTube # <- pytube har stoopid fejl for tiden
from pytubefix import Playlist
from pytubefix.cli import on_progress
from urllib.error import HTTPError

# setting up the directory
download_dir = "/work/pytube test/output/audio_output/interviews_from_JR"


# define playlist from Jonas Risvigs collected interviews
playlist_interviews_JR = Playlist("https://www.youtube.com/watch?v=78MrFjlfo6w&list=PL2XPXGyI_pGOuilA9TpaPIOB34ziju7c_")
print('Number of videos in playlist: %s' % len(playlist_interviews_JR.video_urls))

for video in playlist_interviews_JR.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=download_dir) # donwloading

# downloading masterclass playlist from Jonas Risvig
masterclass_dir = "/work/pytube test/output/audio_output/interviews_from_JR/masterclass"

masterclass = Playlist("https://www.youtube.com/watch?v=vU5Hd7wBxes&list=PL2XPXGyI_pGNyi_4CjbnEeMq1fDwAtKBP")

for video in masterclass.videos:
    audio_stream = video.streams.filter(only_audio=True).first()

    audio_stream.download(output_path=masterclass_dir)



# downloading sole video missing from the playlists
download_audio = "/work/pytube test/output/audio_output"

# Youtube object
yt = YouTube("https://www.youtube.com/watch?v=4mHW0Ff73Eg&ab_channel=P3xDRTV",
on_progress_callback=on_progress)

stream = yt.streams.filter(only_audio=True).first()


stream.download(output_path=download_audio)
