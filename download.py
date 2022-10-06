import os
import fire
import gdown


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_URLS = {
    "human+dog": "https://drive.google.com/file/d/1tIb-kUNEOjPfVF7ltSFvvKgnGOIdz5MB/view?usp=sharing",
}

def download_checkpoint(domain="human+dog"):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    url = CHECKPOINT_URLS[domain]
    output = os.path.join(CHECKPOINT_DIR, f'{domain}.pth')
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)


if __name__ == '__main__':
    fire.Fire(download_checkpoint)