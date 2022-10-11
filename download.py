import os
import click
import gdown


CHECKPOINT_DIR = 'checkpoints'
URLS = {
    'model-human+dog': {
        'url': 'https://drive.google.com/file/d/1tIb-kUNEOjPfVF7ltSFvvKgnGOIdz5MB/view?usp=sharing',
        'filename': 'human+dog.pth',
    },
}

@click.command()
@click.option('--key', default='model-human+dog', help=f"Available keys: [{','.join(URLS.keys())}]")
def download(key):
    if key not in URLS.keys():
        raise ValueError(f"Wrong key value, {key}, available key list is as following: [{','.join(URLS.keys())}]")

    if 'model' in key:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        url = URLS[key]['url']
        filename = URLS[key]['filename']
        output = os.path.join(CHECKPOINT_DIR, filename)
        gdown.download(url=url, output=output, quiet=False, fuzzy=True)
    elif 'data' in key:
        # TODO
        pass
    else:
        raise ValueError(f"Wrong key value, {key}, available key list is as following: [{','.join(URLS.keys())}]")


if __name__ == '__main__':
    download()
