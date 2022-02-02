import subprocess
from glob import glob

files = glob("MP4s/dia*.mp4")

for mp4 in files:
    name = mp4[:-4]
    name = "".join(name.split("\\")[1:])
    command = "ffmpeg -i " + mp4 + " -ab 160k -ac 2 -ar 44100 -vn " + "WAVs/" + name +".wav"
    subprocess.call(command, shell=True)