import os 
import subprocess
import random

rootdir = "generated-samples"

if not os.path.exists("./videos"):
    os.mkdir("./videos")

labels = ['four', 'three', 'two', 'five', 'visual', 'dog', 'go', 
          'nine', 'learn', 'stop', 'on', 'sheila', 'tree', 'follow', 
          'left', 'bird', 'marvin', 'cat', 'no', 'zero', 'six', 'down', 
          'off', 'one', 'bed', 'backward', 'up', 'eight', 'random', 'yes', 
          'forward', 'wow', 'seven', 'right', 'happy', 'house']

targets = ["input", "output"]


def ffmpeg_command(path, name):
    return f"ffmpeg -r 1 -loop 1 -i black_background.png -i {path}.wav -acodec copy -r 1 -shortest -vf scale=1280:720 videos/{name}.avi"

for label in labels:
    selected_samples = random.sample(range(5), 2)
    for target in targets:
        for sample_index in selected_samples:
            name = f"{label}-{target}-{sample_index}"
            path = f"./{rootdir}/{label}/{name}"
            subprocess.call(ffmpeg_command(path, name), shell=True)
