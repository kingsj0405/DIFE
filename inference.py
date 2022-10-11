from pathlib import Path

import click
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from networks.dife import DIFE
from visualizer import GridVisualizer


def center_crop(
    img,
):
    img_crop = img
    B, C, H, W = img_crop.shape
    if H > W:
        x1 = H // 2 - W // 2
        x2 = H // 2 + W // 2
        img_crop = img_crop[:,:,x1:x2,:]
    else:
        y1 = W // 2 - H // 2
        y2 = W // 2 + H // 2
        img_crop = img_crop[:,:,:,y1:y2]
    return img_crop


@click.command()
@click.option('--resume_path', default='checkpoints/human+dog.pth')
@click.option('--image_path', default='demo/human_000001.png')
def main(
    # Network
    dife_dim=16,
    domain_dim=512,
    domain_num=2,
    # Checkpoint
    resume_path='checkpoints/human+dog.pth',
    image_path='demo/human_000001.png',
    # Inference setup
    target_size=96,
):

    net = DIFE(
        dife_dim=dife_dim,
        domain_dim=domain_dim,
        domain_num=domain_num,
    )
    net.cuda()
    net.eval()

    ckpt = torch.load(resume_path)
    net.load_state_dict(ckpt)

    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img = Image.open(image_path)
    img = transform(img)
    img = img[None,:,:,:]
    img = img.cuda()
    img_crop = center_crop(img)
    img_resize = F.interpolate(img_crop, (target_size, target_size))
    out = net(img_resize)

    image_path = Path(image_path)
    vis = GridVisualizer(f"{image_path.parent}/{image_path.stem}_dife.png", (1, 2))
    vis.draw(img_crop[0].detach().cpu(), "image", 1, 1)
    vis.draw(out[0].detach().cpu(), "embedding", 1, 2)
    vis.save()


if __name__ == '__main__':
    main()