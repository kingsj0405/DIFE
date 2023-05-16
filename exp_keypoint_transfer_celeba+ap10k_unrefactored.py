import copy
import os
import shutil

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation_in_style.models.stylegan2.model import Generator
from DVE.test_matching_ours_interspecies import interface_validate_seperate
from src.models.detectron2 import get_detectron2_model_from_file
from src.models.dife import DIFE
from DVE.model.hourglass import (
    HourglassNet,
    ResidualBottleneckPreactivation,
)

import data_loader as module_data
from utils import tps
from utils.tps import spatial_grid_unnormalized, tps_grid
from utils.visualization import norm_range, norm_embedding
from utils.util import dict_coll


def get_embedder():
    embedder = get_detectron2_model_from_file(
        './configs/ssfcse/000_cse_finetune_dve.yaml',
        # './configs/cse/densepose_rcnn_R_101_FPN_DL_soft_animals_finetune_i2m_m2m_16k.yaml',
        './checkpoints/cse/densepose_rcnn_R_101_FPN_DL_soft_animals_finetune_i2m_m2m_16k/model_final.pth',
    )
    return embedder


def get_hourglass(resume_path=None, keep_size=False):
    hg = HourglassNet(
        ResidualBottleneckPreactivation,
        use_group_norm=True,
        num_stacks=1,
        num_output_channels=16,
        output_as_tensor=True,
        keep_size=keep_size,
    )
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise ValueError(f"resume_path not exist, {resume_path}")
        hg.load_state_dict(torch.load(resume_path))
    return hg


def compute_pixel_err(pred_x, pred_y, gt_x, gt_y, imwidth, crop):
    """Compute the pixel error of the corresponding keypoints

    Args:
        pred_x (float): predicted x-coordinate for keypoint
        pred_y (float): predicted y-coordinate for keypoint
        gt_x (float): ground truth x-coordinate for keypoint
        gt_y (float): ground truth y-coordinate for keypoint
        imwidth (int): the width of the image (pixels)
        crop (int): the size of the crop from the boundary (pixels)

    Returns:
        (float) pixel error
    NOTE: To account for different input sizes, we scale all distances as
    though they occured in pixel space for a 70x70 (post-crop) image
    (this was used in the original version of the model so allows
    for comparison).
    """
    canonical_sz = 70
    scale = canonical_sz / (imwidth - 2 * crop)
    pred_x = pred_x * scale
    pred_y = pred_y * scale
    gt_x = gt_x * scale
    gt_y = gt_y * scale
    return np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)


def plt_save(image, filename, kp_gt=None, kp_pred=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.axis('off')

    ax.imshow(image)
    m_list = ['o','X','s','X','s','o','s','o','X']
    c_list = ['r','m','b','r','m','b','r','m','b']
    if kp_gt is not None:
        N, _ = kp_gt.shape
        for ki in range(N):
            ax.scatter(kp_gt[ki, 0], kp_gt[ki, 1], c=c_list[ki], s=500, marker=m_list[ki])
    if kp_pred is not None:
        N, _ = kp_pred.shape
        for ki in range(N):
            ax.scatter(kp_pred[ki, 0], kp_pred[ki, 1], c=c_list[ki], s=500, marker=m_list[ki])

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()


def find_descriptor(x, y, source_descs, target_descs, stride):
    C, H, W = source_descs.shape
    x = int(np.round(x / stride))
    y = int(np.round(y / stride))
    x = min(W - 1, max(x, 0))
    y = min(H - 1, max(y, 0))
    query_desc = source_descs[:, y, x]
    corr = torch.matmul(query_desc.reshape(-1, C), target_descs.reshape(C, H * W))
    maxidx = corr.argmax()
    grid = spatial_grid_unnormalized(H, W).reshape(-1, 2) * stride
    x, y = grid[maxidx]
    return x.item(), y.item()


def evaluation(
    model_list,
    data_root_human,
    data_root_animal,
    source,
    target,
    vis_dir,
):
    device = 'cuda:0'
    imwidth = 96
    crop = 0

    # Want explicit pair warper
    warp_kwargs = dict(
        warpsd_all=0.001 * .5,
        warpsd_subset=0.01 * .5,
        transsd=0.1 * .5,
        scalesd=0.1 * .5,
        rotsd=5 * .5,
        im1_multiplier=1,
        im1_multiplier_aff=1,
    )
    warper = tps.Warper(imwidth, imwidth, **warp_kwargs)
    eval_data = 'InterspeciesVal'
    constructor = getattr(module_data, eval_data)

    kwargs = dict()
    # if source == "human":
    #    kwargs.update(
    #        dict(
    #            source_img_dir=f"{data_root_human}/crop_images",
    #            source_annotation=f"{data_root_human}/crop_annotations/wflw-test-crop.json",
    #        )
    #    )
    # else:
    #    kwargs.update(
    #        dict(
    #            source_img_dir=f"{data_root_animal}/crop_images/{source}",
    #            source_annotation=f"{data_root_animal}/crop_annotations/animalweb-test-{source}-crop.json",
    #        )
    #    )
    
    # kwargs.update(
    #    dict(
    #        animal_img_dir=f"{data_root_animal}/crop_images/{target}",
    #        animal_annotation=f"{data_root_animal}/crop_annotations/animalweb-test-{target}-crop.json",
    #    )
    # )
    # handle the case of the MAFL split, which by default will evaluate on Celeba
    kwargs = {"val_split": "mafl"} if eval_data == "CelebAPrunedAligned_MAFLVal" else {}
    kwargs.update(
        dict(
            animal_img_dir=f"data/ap-10k/img_{target}_crop",
            animal_annotation=f"data/ap-10k/annotations/ap10k-test-{target}-crop.json",
        )
    )
    if source != "human":
        kwargs.update(
            dict(
                source_img_dir=f"data/ap-10k/img_{source}_crop",
                source_annotation=f"data/ap-10k/annotations/ap10k-test-{source}-crop.json"
            )
        )
    val_dataset = constructor(
        train=False,
        pair_warper=warper,
        use_keypoints=True,
        imwidth=imwidth,
        crop=crop,
        root="data/celeba",
        **kwargs,
    )
    # NOTE: Since the matching is performed with pairs, we fix the ordering and then
    # use all pairs for datasets with even numbers of images, and all but one for
    # datasets that have odd numbers of images (via drop_last=True)
    data_loader = DataLoader(val_dataset, batch_size=2, collate_fn=dict_coll,
                             shuffle=False, drop_last=True)
    os.makedirs(f'{vis_dir}', exist_ok=True)

    for model_id, model in enumerate(model_list):
        model = copy.deepcopy(model)
        model = model.to(device)
        model.eval()

        cros_errs = []

        torch.manual_seed(0)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                data, meta = batch["data"], batch["meta"]

                data = data.to(device)
                output = model(data)

                im_src = data[0].cpu()
                im_tgt = data[2].cpu()

                _, _, imW = im_src.shape
                _, _, _, W = output.shape
                stride = imW / W

                kp_src = meta['kp1'][0,:3]
                kp_tgt = meta['kp3'][0]
                # kp_src = meta['kp1'][0,:9]
                # kp_tgt = meta['kp3'][0]

                fsrc = output[0].cpu()
                fcros = output[2].cpu()

                fig = plt.figure()  # a new figure window
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)

                ax1.imshow(norm_range(im_src).permute(1, 2, 0))
                ax2.imshow(norm_range(im_tgt).permute(1, 2, 0))

                ax1.scatter(kp_src[:, 0], kp_src[:, 1], c='g')
                ax2.scatter(kp_tgt[:, 0], kp_tgt[:, 1], c='g')

                src_kps = kp_src.cuda()
                src_kps = src_kps[None,:,:]
                src_kps = src_kps.permute(0,2,1)

                cros_kps = []
                for ki, kp in enumerate(kp_src):
                    x, y = np.array(kp)
                    gt_cros_x, gt_cros_y = np.array(kp_tgt[ki])
                    cros_x, cros_y = find_descriptor(x, y, fsrc, fcros, stride)
                    cros_kps.append([cros_x, cros_y])
                    err = compute_pixel_err(
                        pred_x=cros_x,
                        pred_y=cros_y,
                        gt_x=gt_cros_x,
                        gt_y=gt_cros_y,
                        imwidth=imwidth,
                        crop=crop,
                    )
                    cros_errs.append(err)

                    ax2.scatter(cros_x, cros_y, c='b')
                cros_kps = np.array(cros_kps)

                ax1.axis('off')
                ax2.axis('off')
                fig.savefig(f'{vis_dir}/{i:03d}_model{model_id}.png')
                plt.close()

                plt_save(
                    norm_range(im_src).permute(1, 2, 0),
                    f'{vis_dir}/{i:03d}_source.png',
                )
                plt_save(
                    norm_range(im_src).permute(1, 2, 0),
                    f'{vis_dir}/{i:03d}_source_kp.png',
                    kp_src,
                )
                plt_save(
                    norm_range(im_tgt).permute(1, 2, 0),
                    f'{vis_dir}/{i:03d}_target.png',
                )
                plt_save(
                    norm_range(im_tgt).permute(1, 2, 0),
                    f'{vis_dir}/{i:03d}_target_kp_model{model_id}.png',
                    None,#kp_cros,
                    np.array(cros_kps),
                )
        print(f'[INFO] model: {model_id} domain {source}+{target} cros_err: {np.mean(cros_errs)}')


def main():
    option = edict(
        dve_resume_path = "checkpoints/dve/model_best_iter_0011300.pth",
        ours_resume_path = {
            "human+dog": "checkpoints/dife/celeba+ap10k/dife_human+dog_0004000.pth",
            "human+cat": "checkpoints/dife/celeba+ap10k/dife_human+cat_0005700.pth",
            "dog+cat": "checkpoints/dife/celeba+ap10k/dife_dog+cat_0008600.pth",
            "human+wild": "checkpoints/dife/celeba+ap10k/dife_human+cat_0005700.pth",
        },
        save_dir = "output_exp/exp_keypoint_transfer/celeba+ap10k",
        test_data_dir = dict(
            human="./data/wflw",
            animal="./data/AnimalWeb",
            # human="./data/celeba",
            # animal="./data/ap-10k",
        ),
    )

    if os.path.exists(option.save_dir):
        shutil.rmtree(option.save_dir)
    os.makedirs(option.save_dir, exist_ok=True)

    for data_type in ["human+dog", "human+cat", "dog+cat", "human+wild"]:
        source, target = data_type.split('+')

        densepose_cse = get_embedder()
        densepose_cse.cuda()
        densepose_cse.eval()

        net_dve = get_hourglass(option.dve_resume_path)
        net_dve.cuda()
        net_dve.eval()

        net = get_hourglass(option.ours_resume_path[data_type], keep_size=True)
        net.cuda()
        net.eval()
        
        evaluation(
            [densepose_cse, net_dve, net],
            data_root_human=option.test_data_dir['human'],
            data_root_animal=option.test_data_dir['animal'],
            vis_dir=f"{option.save_dir}/{data_type}",
            source=source,
            target=target,
        )


if __name__ == '__main__':
    main()
