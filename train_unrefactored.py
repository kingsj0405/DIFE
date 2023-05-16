"""
This code will not work now.
The code is uploaded just for sharing the pseudo code of training.
I'll try to refactor ASAP.

CUDA_VISIBLE_DEVICES=0 python scripts/train_unrefactored.py
"""
import os
import shutil
import yaml
import pprint

import fire
import wandb
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.datasets import CseSsfDataset
from src.models.detectron2 import get_detectron2_model_from_file
from src.models.stylegan_seg import get_generator
from src.models.dife import DIFE
from src.utils.visualize import Visualizer
from scripts.test_ours_interspecies import (
    evaluation,
)
from src.utils.train_util import (
    BestLossSavor,
    EarlyStop,
)
from src.utils.misc import (
    remove_randomness,
    log,
)
from src.losses import (
    distillation_loss,
    cross_matching_loss_weighted,
)


def main(config_file_path="./configs/dife/human_dog.yaml"):
    with open(config_file_path) as f:
        config = yaml.full_load(f)
    config = edict(config)

    domains = config.data_type.split('+')
    domain_num = len(domains)
    cse_config_path = config.cse_config_path
    cse_weight_path = config.cse_weight_path
    sg2_weight_path = config.sg2_weight_path
    for d in domains:
        sg2_weight_path[d] = sg2_weight_path[d]
    resume_path = config.resume_path
    test_data_dir = config.test_data_dir
    test_data_dir['human'] = test_data_dir['human']
    test_data_dir['animal'] = test_data_dir['animal']
    save_dir = config.save_dir
    log_file_path = f'{config.save_dir}/train_ours_interspecies.log'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    log(log_file_path, f'config :{pprint.pformat(config)}')

    transform_train = T.Compose([
        T.RandomApply(torch.nn.ModuleList([
            T.ColorJitter(brightness=.1, hue=.1),
        ]), p=0.7),
    ])

    densepose_cse = get_detectron2_model_from_file(
        cse_config_path,
        cse_weight_path,
    )
    densepose_cse.cuda()
    densepose_cse.train()

    sg2 = {}
    mean_latent = {}
    for d in domains:
        sg2[d] = get_generator(d, sg2_weight_path[d])
        mean_latent[d] = sg2[d].mean_latent(4096)
    trunsourceion = 0.7
    img_size = 96

    net = DIFE(
        transformer_input_dim=16,
        transformer_output_dim=config.domain_embedding_dim,
        transformer_domain_num=domain_num,
    )
    net.cuda()
    net.train()

    net.resume(resume_path)

    optimizer = torch.optim.Adam(
        list(net.parameters()),
        lr=config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.reduce_patience_cnt)
    ea = EarlyStop(patience=config.early_stop_cnt)
    bss = BestLossSavor()

    wandb.init(project="nips2022", entity="kingsj0405", name=config.run_name)
    wandb.config.update(config)

    net = torch.nn.DataParallel(net)
    for i in tqdm(range(config.max_iter)):

        remove_randomness(config.random_seed + i)

        utilize_pseudo_pair = i > config.init_period_without_pseudo_pair
        synthesize_pseudo_pair = (utilize_pseudo_pair and i % config.synthesize_period  == 0) or (i == (config.init_period_without_pseudo_pair + 1))
        if utilize_pseudo_pair:
            origin_batch_size = (config.batch_size) // (domain_num  * (1 + domain_num))
            origin_batch_size -= 1
            origin_batch_size = max(origin_batch_size, 1)
        else:
            origin_batch_size = (config.batch_size) // (domain_num)

        # (1) Generate general sample
        with torch.no_grad():
            sample_z = torch.randn(origin_batch_size, 512).cuda()
            style = {}
            img = {}
            sgf = {}
            for d in domains:
                style[d] = sg2[d].style(sample_z)
                style[d] = mean_latent[d] + trunsourceion * (style[d] - mean_latent[d])
                img_d, _, sgf_d = sg2[d](
                    [style[d]],
                    input_is_latent=True,
                    randomize_noise=False,
                    feature_layer_number_to_return=config.layer_num,
                )
                img_d = F.interpolate(
                    img_d,
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=True,
                )
                img_d = img_d.clamp(min=-1.0, max=1.0)
                sgf_d = F.interpolate(sgf_d, (img_size, img_size))
                img[d] = img_d
                sgf[d] = sgf_d

            if synthesize_pseudo_pair:
                # (2) Generate paired data
                img_ori = torch.cat(list(img.values()), dim=0)
                out_net = net(img_ori)
                img_paired_ori_nowarp = {}
                img_paired_gen_nowarp = {}
                sgf_paired_gen_nowarp = {}
                for j, d in enumerate(domains):
                    # Generate cross domain paired data
                    out_t = net(out_net, f'd{j}')
                    out_t = F.interpolate(out_t, (config.sgf_size, config.sgf_size))
                    img_paired_gen_d, sgf_paired_gen_d = sg2[d].get_image_based_on_feature(
                        [torch.cat([style[d]] * domain_num, dim=0)],
                        feature=out_t,
                        feature_layer_number_to_return=config.layer_num,
                        max_iter=config.utilize_pseudo_pair_max_iter,
                        threshold=config.utilize_pseudo_pair_threshold,
                    )
                    img_paired_gen_d = F.interpolate(img_paired_gen_d, (img_size, img_size))
                    sgf_paired_gen_d = F.interpolate(sgf_paired_gen_d, (img_size, img_size))
                    # Set domain datas
                    img_paired_ori_nowarp[d] = img_ori
                    img_paired_gen_nowarp[d] = img_paired_gen_d
                    sgf_paired_gen_nowarp[d] = sgf_paired_gen_d
            
            # Merge
            img_list = []
            sgf_list = []
            for j, d in enumerate(domains):
                img_list.append(img[d])
                sgf_list.append(sgf[d])
                if utilize_pseudo_pair:
                    img_list.append(img_paired_gen_nowarp[d])
                    sgf_list.append(sgf_paired_gen_nowarp[d])
            img_merge = torch.cat(img_list, dim=0)
            sgf_merge = torch.cat(sgf_list, dim=0)
            # Augmentation
            img_warp, sgf_warp = CseSsfDataset.random_affine_warp([img_merge, sgf_merge])
            img_warp = transform_train(img_warp)

            if utilize_pseudo_pair:
                img_paired_ori = {}
                img_paired_gen = {}
                out_net_pair_ori = {}
                for j, d in enumerate(domains):
                    # Augmentation
                    img_warp_d, img_paired_gen_warp_d = CseSsfDataset.random_affine_warp([img_paired_ori_nowarp[d], img_paired_gen_nowarp[d]])
                    img_warp_d = transform_train(img_warp_d)
                    img_paired_gen_warp_d = transform_train(img_paired_gen_warp_d)
                    img_paired_ori[d] = img_warp_d
                    img_paired_gen[d] = img_paired_gen_warp_d
                    # Prepare Stop Gradiented Interspecies Embedding
                    out_net_pair_ori[d] = net(img_paired_ori[d])
                    out_net_pair_ori[d] = F.interpolate(out_net_pair_ori[d], (96//2, 96//2))
        
        out_cse_warp = densepose_cse(img_warp)
        out_net_warp = net(img_warp)
        B, _, _, _ = out_net_warp.shape
        out_sgf_warp = []
        for j, d in enumerate(domains):
            out_t_d = net(
                out_net_warp[
                    B // domain_num * j :
                    B // domain_num * (j + 1)
                ],
                f'd{j}',
            )
            out_sgf_warp.append(out_t_d)
        out_sgf_warp = torch.cat(out_sgf_warp, dim=0)
        if utilize_pseudo_pair:
            loss_cse_dist = config.lambda_cse_dist * distillation_loss(
                out_net_warp[0::(domain_num+1)],
                out_cse_warp[0::(domain_num+1)],
            )
            loss_sgf_dist = config.lambda_sgf_dist * distillation_loss(
                out_sgf_warp[0::(domain_num+1)],
                sgf_warp[0::(domain_num+1)],
            )
        else:
            loss_cse_dist = config.lambda_cse_dist * distillation_loss(out_net_warp, out_cse_warp)
            loss_sgf_dist = config.lambda_sgf_dist * distillation_loss(out_sgf_warp, sgf_warp)
        loss = loss_cse_dist + loss_sgf_dist

        if utilize_pseudo_pair:
            with torch.enable_grad():
                loss_match = 0
                out_net_pair_gen = {}
                for d in domains:
                    out_net_pair_gen[d] = net(img_paired_gen[d])
                    out_net_pair_gen[d] = F.interpolate(out_net_pair_gen[d], (96//2, 96//2))
                loss_match = config.lambda_cross * cross_matching_loss_weighted(
                    torch.cat(list(out_net_pair_ori.values()), dim=0),
                    torch.cat(list(out_net_pair_gen.values()), dim=0),
                )
                loss += loss_match
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #############################################################################################
        # Visualize, Train Meta
        #############################################################################################
        wandb.log({
            'loss': loss.cpu().detach(),
            'loss_cse_dist': loss_cse_dist.cpu().detach(),
            'loss_sgf_dist': loss_sgf_dist.cpu().detach(),
            'lr': scheduler.optimizer.param_groups[0]['lr'],
        }, step=i)

        if utilize_pseudo_pair:
            wandb.log({
                'loss_match': loss_match.cpu().detach(),
            }, step=i)

        if i % config.vis_period == 0:

            vis_cnt = origin_batch_size * domain_num
            vis_row = vis_cnt * (1 + (2 * domain_num * utilize_pseudo_pair))
            vis = Visualizer(f"{save_dir}/vis_iter_{i:07d}.png", (vis_row, 5))

            cur_row = 0
            j_index = 0
            for j in range(vis_cnt):
                vis.draw(img_warp[j_index].detach().cpu(), "image_tensor_chw", cur_row + 1, 1)
                vis.draw(out_cse_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 2)
                vis.draw(sgf_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 3)
                vis.draw(out_net_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 4)
                vis.draw(out_sgf_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 5)
                j_index += 1
                cur_row += 1

                if utilize_pseudo_pair:
                    for d in domains:
                        vis.draw(img_warp[j_index].detach().cpu(), "image_tensor_chw", cur_row + 1, 1)
                        vis.draw(out_cse_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 2)
                        vis.draw(sgf_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 3)
                        vis.draw(out_net_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 4)
                        vis.draw(out_sgf_warp[j_index].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 5)
                        j_index += 1
                        cur_row += 1

                    for d in domains:
                        vis.draw(img_paired_ori[d][j].detach().cpu(), "image_tensor_chw", cur_row + 1, 2)
                        vis.draw(out_net_pair_ori[d][j].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 3)
                        vis.draw(img_paired_gen[d][j].detach().cpu(), "image_tensor_chw", cur_row + 1, 4)
                        vis.draw(out_net_pair_gen[d][j].detach().cpu(), "embedding_tensor_chw", cur_row + 1, 5)
                        cur_row += 1
            
            vis.save()

            if i > config.vis_period * config.vis_cnt:
                os.remove(f"{save_dir}/vis_iter_{i - (config.vis_period * config.vis_cnt):07d}.png")

        if i % config.val_period == 0:
            same_errs = 0
            diff_errs = 0
            cros_errs = 0
            # Calculate loss valid
            with torch.no_grad():
                if config.data_type == "human+dog+cat":
                    source_d = ["human", "human", "dog"]
                    target_d = ["dog", "cat", "cat"]
                else:
                    source_d = [domains[0]]
                    target_d = [domains[1]]
                    cnt_d = []
                for di in range(len(source_d)):
                    same_err, diff_err, cros_err, cnt = evaluation(
                        net,
                        data_root_human=test_data_dir['human'],
                        data_root_animal=test_data_dir['animal'],
                        source=source_d[di],
                        target=target_d[di],
                        vis_dir=f"{save_dir}/exp_keypoint_transfer_interspecies",
                    )
                    log(log_file_path, f"[INFO] {source_d[di]} to {target_d[di]}: cros_err {cros_err}")
                    same_errs += same_err * cnt
                    diff_errs += diff_err * cnt
                    cros_errs += cros_err * cnt
                    cnt_d.append(cnt)
                same_errs /= sum(cnt_d)
                diff_errs /= sum(cnt_d)
                cros_errs /= sum(cnt_d)
            valid_accuracy = cros_errs
            # Utilize valid val
            bss.check(valid_accuracy, net)
            wandb.log({
                "best_valid_loss": bss.best_loss,
                "same_errs": same_errs,
                "diff_errs": diff_errs,
                "cros_errs": cros_errs,
            }, step=i)
            scheduler.step(valid_accuracy)
            early_stop = ea.check(valid_accuracy)
            if early_stop: break
        
        if i % config.save_period == 0:
            torch.save(bss.get_best_model().module.state_dict(), f'{save_dir}/model_best_iter_{i:07d}.pth')
            torch.save(net.module.state_dict(), f'{save_dir}/model_last_iter_{i:07d}.pth')
            if i > 0: os.remove(f"{save_dir}/model_best_iter_{i - config.save_period:07d}.pth")


if __name__ == '__main__':
    fire.Fire(main)
