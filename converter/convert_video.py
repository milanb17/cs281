import skvideo.io
import skvideo.datasets
import cv2
import pickle
import numpy as np
import os

def generate_video_frames(video):
    videogen = skvideo.io.vreader(video, inputdict = {'-r': "2"}, outputdict = {'-r': "2"})
    for frame in videogen:
        res = cv2.resize(frame, dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)
        yield np.transpose(res)

# Build arrays.
trailers_normal = list()
trailers_chunked = list()
scores = list()

for filename in os.listdir("../trailers"):
    try:
        print(filename)
        # arr.shape = (100, 3, 64, 64)
        arr = np.array(list(generate_video_frames(f"../trailers/{filename}")))[:100]
        # If the number of frames is not 100, ignore this video.
        if len(arr) != 100:
            continue

        arr_chunked = np.swapaxes(np.array(np.split(arr, 10)), 1, 2)

        trailers_normal.append(arr)
        trailers_chunked.append(arr_chunked)
        scores.append(int(filename.split("--")[1][0:2]) / 100)
    except:
        pass

pickle.dump(np.array(trailers_normal), open("trailers_normal.p", "wb"))
pickle.dump(np.array(trailers_chunked), open("trailers_chunked.p", "wb"))
pickle.dump(np.array(scores), open("scores.p", "wb"))
