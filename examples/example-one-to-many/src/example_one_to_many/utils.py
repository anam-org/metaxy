import random


def split_video_into_chunks(path: str) -> list[str]:
    # pretend we load the video and split it to chunks
    num_chunks = random.randint(3, 5)
    return [f"{path}_{i}" for i in range(num_chunks)]
