import os
import shutil
import pickle

import torch
import torch.nn as nn
import fire
from tqdm import tqdm
from sklearn.cluster import KMeans

from segmentation_in_style.models.stylegan2.model import Generator
from src.models.dife import DIFE


def get_generator(type):
    if type == "human":
        g_ema = Generator(1024, 512, 8)
        g_ema.load_state_dict(torch.load('./checkpoints/stylegan_seg/stylegan2-ffhq-config-f.pt')["g_ema"], strict=False)
        g_ema.eval()
        g_ema = g_ema.cuda()
    elif type == "dog":
        g_ema = Generator(512, 512, 8)
        g_ema.load_state_dict(torch.load('./checkpoints/stylegan_seg/animals_dog_ada.pth'), strict=True)
        g_ema.eval()
        g_ema = g_ema.cuda()
    elif type == "cat":
        g_ema = Generator(512, 512, 8)
        g_ema.load_state_dict(torch.load('./checkpoints/stylegan_seg/animals_cat_ada.pth'), strict=True)
        g_ema.eval()
        g_ema = g_ema.cuda()
    elif type == "wild":
        g_ema = Generator(512, 512, 8)
        g_ema.load_state_dict(torch.load('./checkpoints/stylegan_seg/animals_wild_ada.pth'), strict=True)
        g_ema.eval()
        g_ema = g_ema.cuda()
    else:
        raise ValueError(f"type is wrong, type: {type}")
    return g_ema


def main(n_colors=9):
    resume_path = "outputs/wflw+animalweb/human+dog/003/model_best_iter_0012000.pth"
    data_type = "human+dog"
    count = 100
    image_size = 96
    save_dir = f"output_exp/exp_face_parsing/{data_type}_{n_colors}"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    net = DIFE(16, 512, 2)
    net.resume(resume_path)
    net.cuda()
    net.eval()

    source, target = data_type.split('+')
    sg2_source = get_generator(source)
    sg2_target = get_generator(target)
    mean_latent_source = sg2_source.mean_latent(4096)
    mean_latent_target = sg2_target.mean_latent(4096)
    truncation = 0.7

    features = None
    imgs = None

    with torch.no_grad():
        for i in tqdm(range(count)):
            if i < count // 2:
                sample_z = torch.randn(1, 512).cuda()
                style = sg2_source.style(sample_z)
                style = mean_latent_source + truncation * (style - mean_latent_source)
                img, _, _ = sg2_source([style], input_is_latent=True, randomize_noise=False, feature_layer_number_to_return=7)
            else:
                sample_z = torch.randn(1, 512).cuda()
                style = sg2_target.style(sample_z)
                style = mean_latent_target + truncation * (style - mean_latent_target)
                img, _, _ = sg2_target([style], input_is_latent=True, randomize_noise=False, feature_layer_number_to_return=7)
            img = nn.functional.upsample(
                img,
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=True,
            ).clamp(min=-1.0, max=1.0).detach()
            out = net.forward_hg(img)
            out = nn.functional.upsample(
                out,
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=True,
            ).detach()
            if imgs is None:
                imgs = img.cpu()
            else:
                aditional_imgs = img.cpu()
                imgs = torch.cat((imgs, aditional_imgs), axis=0)
            if features is None:
                features = out.cpu()
            else:
                additional_features = out.cpu()
                features = torch.cat((features, additional_features), axis=0)

    print(f"[INFO] features: {features.shape}")
    _, C, _, _ = features.shape

    features_new = features.permute(0, 2, 3, 1).reshape(-1, C)

    arr = features_new.detach().cpu().numpy()#dist.detach().cpu().numpy().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_colors, random_state=903).fit(arr)
    with open(f"{save_dir}/kmeans_cluster.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    labels = kmeans.labels_
    labels_spatial = labels.reshape(features.shape[0], features.shape[2], features.shape[3])

    from src.utils.visualize import Visualizer
    for i in tqdm(range(count // 2)):
        image_path = f"{save_dir}/{i:06d}.png"
        vis = Visualizer(
            save_path=image_path,
            grid=(3, 2),
        )
        vis.draw(imgs[i], "image_tensor_chw", 1, 1)
        vis.draw(imgs[i + count // 2], "image_tensor_chw", 1, 2)
        vis.draw(features[i], "embedding_tensor_chw", 2, 1)
        vis.draw(features[i + count // 2], "embedding_tensor_chw", 2, 2)
        vis.draw(labels_spatial[i], "cvimage", 3, 1)
        vis.draw(labels_spatial[i + count // 2], "cvimage", 3, 2)
        vis.save()

        vis = Visualizer(save_path=f"{save_dir}/{i:06d}_1_src_img.png")
        vis.draw(imgs[i], "image_tensor_chw", 1, 1)
        vis.save()

        vis = Visualizer(save_path=f"{save_dir}/{i:06d}_2_tgt_img.png")
        vis.draw(imgs[i + count // 2], "image_tensor_chw", 1, 1)
        vis.save()

        vis = Visualizer(save_path=f"{save_dir}/{i:06d}_3_src_feat.png")
        vis.draw(features[i], "embedding_tensor_chw", 1, 1)
        vis.save()

        vis = Visualizer(save_path=f"{save_dir}/{i:06d}_4_tgt_feat.png")
        vis.draw(features[i + count // 2], "embedding_tensor_chw", 1, 1)
        vis.save()

        vis = Visualizer(save_path=f"{save_dir}/{i:06d}_5_src_parse.png")
        vis.draw(labels_spatial[i], "cvimage", 1, 1)
        vis.save()

        vis = Visualizer(save_path=f"{save_dir}/{i:06d}_6_src_parse.png")
        vis.draw(labels_spatial[i + count // 2], "cvimage", 1, 1)
        vis.save()


if __name__ == "__main__":
    fire.Fire(main)
