"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_single_mlm import SingleMlm
from models.vit import interpolate_pos_embed
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, mlm_single_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer


name_to_model = {
    'SingleMlm': SingleMlm,
}


def train_epoch(args, model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config, tokenizer=None):
    assert args.mode in ['train', 'both']
    # train
    model.train()

    save_cases, f_case = tokenizer is not None, None
    if save_cases:
        f_case = open(os.path.join(config['output_path'], 'logs_train',
                                   f'log_case_train_{epoch}.txt'), 'w', encoding='utf-8')

    metric_logger = utils.MetricLogger(f_path=os.path.join(config['output_path'], "log_metric.txt"), delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_rec', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_plain', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_tra_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('correct_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    metric_logger.add_meter('instance_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    for k in config['topk']:
        metric_logger.add_meter(f'hit_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
        metric_logger.add_meter(f'rank_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))
        metric_logger.add_meter(f'rank_instance_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    data_idx = 0

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, mask_ori_images, input_ids, attn_masks, labels, plain_labels, pos_ids, type_ids, lengths,
            book_orders, mask_ids, mask_img_ids, mask_chs) \
            in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        mask_ori_images = mask_ori_images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        plain_labels = plain_labels.to(device, non_blocking=True)
        pos_ids = pos_ids.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        mask_ids = mask_ids.to(device, non_blocking=True)
        mask_img_ids = mask_img_ids.to(device, non_blocking=True)

        total_loss, loss_mlm, loss_rec, loss_plain, loss_tra_mlm, correct_num, instance_num, ori_inputs, \
            correct_chars, wrong_chars, rank_correct_num, rank_instance_num, hit_correct, topk_ids, topk_scores = \
            model(images, mask_ori_images, input_ids, attn_masks, labels, plain_labels, pos_ids, type_ids, lengths,
                  mask_ids, mask_img_ids, mask_chs, 'train')

        total_loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_rec=loss_rec.item())
        metric_logger.update(loss_plain=loss_plain.item())
        metric_logger.update(loss_tra_mlm=loss_tra_mlm.item())
        metric_logger.update(correct_num=int(correct_num))
        metric_logger.update(instance_num=int(instance_num))
        update_map = {}
        for k in config['topk']:
            update_map[f'hit_correct_{k}'] = hit_correct[k]
            update_map[f'rank_correct_{k}'] = rank_correct_num[k]
            update_map[f'rank_instance_{k}'] = rank_instance_num[k]
        metric_logger.update(**update_map)

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        if save_cases:
            for ch, idx in correct_chars:
                f_case.write(f'{tokenizer.convert_ids_to_tokens(ch)}\t{str(idx)}\n')
            f_case.write('\n')
            wrong_chars = [f'{tokenizer.convert_ids_to_tokens(ch)} {tokenizer.convert_ids_to_tokens(wch)} {str(idx)}'
                           for ch, wch, idx in wrong_chars]
            f_case.write('Wrong: ' + str(wrong_chars) + '\n\n')
            max_k = max(config['topk'])
            assert len(book_orders) == len(ori_inputs) == len(topk_ids) == len(topk_scores)
            for sent, book_order, topk_id, topk_score in zip(ori_inputs, book_orders, topk_ids, topk_scores):
                f_case.write(str(data_idx) + '\t' +
                             str(tokenizer.convert_ids_to_tokens(sent)) + '\t' + book_order + '\n')
                topk_chs, topk_pairs = tokenizer.convert_ids_to_tokens(topk_id), []
                for ch, score in zip(topk_chs, topk_score):
                    topk_pairs.append((ch, round(score, 4)))
                f_case.write(f'Top{max_k}: ' + str(topk_pairs) + '\n')
                data_idx += 1
            f_case.write('------------------------------\n\n')
        f_case.flush()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_default_metric()
    meters = metric_logger.meters
    res = {k: meter.metric_fmt.format(meter.default_metric) for k, meter in meters.items()}
    res['global_accuracy'] = round(100 * meters['correct_num'].total / meters['instance_num'].total, 2)
    for k in config['topk']:
        res[f'global_hit_{k}'] = round(100 * meters[f'hit_correct_{k}'].total / meters['instance_num'].total, 2)
        res[f'global_rank_acc_{k}'] = round(
            100 * meters[f'rank_correct_{k}'].total / meters[f'rank_instance_{k}'].total, 2)

    if save_cases:
        f_case.close()

    return res


def train(args, config, model, train_loader, test_loader=None, tokenizer=None):
    device = torch.device(args.device)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    cur_max_global_acc = 0.0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        elif 'visual_encoder.pos_embed' in state_dict:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        msg = model.load_state_dict(state_dict, strict=False)
        cur_max_global_acc = checkpoint['global_accuracy']
        print('loading complete:', msg)
        print('load checkpoint from %s' % args.checkpoint)
        print('baseline accuracy ' + str(cur_max_global_acc))
    elif args.load_cross:
        if config['modality'] == 'cross':
            assert args.text_checkpoint != '' and args.image_checkpoint != ''
            total_dict = torch.load(args.text_checkpoint, map_location='cpu')['model']
            image_cp = torch.load(args.image_checkpoint, map_location='cpu')['model']
            total_dict.update(image_cp)
            for key in [k for k in total_dict.keys() if k.startswith('classification_head')]:
                del total_dict[key]
            if config['image_reconstruct_factor'] <= 0:
                for key in [k for k in total_dict.keys() if k.startswith('reconstruct_')]:
                    del total_dict[key]
            else:
                print('reconstruct module loaded ......')
        elif config['modality'] == 'text':
            assert args.text_checkpoint != ''
            total_dict = torch.load(args.text_checkpoint, map_location='cpu')['model']
        else:
            assert args.image_checkpoint != ''
            total_dict = torch.load(args.image_checkpoint, map_location='cpu')['model']
            for key in [k for k in total_dict.keys() if k.startswith('classification_head')]:
                del total_dict[key]
            if config['image_reconstruct_factor'] <= 0:
                for key in [k for k in total_dict.keys() if k.startswith('reconstruct_')]:
                    del total_dict[key]
            else:
                print('reconstruct module loaded ......')
            # initialize text encoder normally
            text_null_dict = {}
            for key, val in model.text_encoder.state_dict().items():
                text_null_dict['text_encoder.' + key] = val
            # initialize MLP normally
            if hasattr(model, 'middle_mlp'):
                for key, val in model.middle_mlp.state_dict().items():
                    text_null_dict['middle_mlp.' + key] = val
            total_dict.update(text_null_dict)
        model.load_state_dict(total_dict)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train_epoch(args, model, train_loader, optimizer,
                                  epoch, warmup_steps, device, lr_scheduler, config, tokenizer=tokenizer)
        # for validation
        test_stats = None
        if test_loader is not None:
            test_stats = test_epoch(args, model, test_loader, epoch, device, config, tokenizer=tokenizer)
            if test_stats['global_accuracy'] > cur_max_global_acc:
                cur_max_global_acc = test_stats['global_accuracy']
                save_flag = True
                remove_flag = True
            else:
                save_flag = False
                remove_flag = False
        else:
            # train mode, save the newest
            save_flag = True
            remove_flag = False
        save_flag |= args.save_all
        remove_flag &= not args.save_all

        if utils.is_main_process():
            if remove_flag:
                os.system(f'rm -rf {config["output_path"]}/*.pth')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            if save_flag:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'global_accuracy': cur_max_global_acc,
                }
                torch.save(save_obj, os.path.join(config['output_path'], 'checkpoint_%02d.pth' % epoch))

            print(log_stats)
            with open(os.path.join(config['output_path'], "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if test_stats is not None:
                log_stats = {**{f'valid_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}
                print(log_stats)
                with open(os.path.join(config['output_path'], "log_valid.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def test_epoch(args, model, data_loader, epoch, device, config, tokenizer=None):
    assert args.mode in ['test', 'both']
    model.eval()
    mod = 'valid' if args.mode == 'both' else 'test'

    save_cases, f_case = tokenizer is not None, None
    if save_cases:
        f_case = open(os.path.join(config['output_path'], f'logs_{mod}',
                                   f'log_case_{mod}_{epoch}.txt'), 'w', encoding='utf-8')

    metric_logger = utils.MetricLogger(
        f_path=os.path.join(config['output_path'], f"log_{mod}_metric.txt"), delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_rec', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_plain', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_tra_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('correct_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    metric_logger.add_meter('instance_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    for k in config['topk']:
        metric_logger.add_meter(f'hit_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
        metric_logger.add_meter(f'rank_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))
        metric_logger.add_meter(f'rank_instance_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))

    header = 'Valid Epoch: [{}]'.format(epoch)
    print_freq = 10
    data_idx = 0

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, mask_ori_images, input_ids, attn_masks, labels, plain_labels, pos_ids, type_ids, lengths,
            book_orders, mask_ids, mask_img_ids, mask_chs) \
            in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        images = images.to(device, non_blocking=True)
        mask_ori_images = mask_ori_images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        plain_labels = plain_labels.to(device, non_blocking=True)
        pos_ids = pos_ids.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        mask_ids = mask_ids.to(device, non_blocking=True)
        mask_img_ids = mask_img_ids.to(device, non_blocking=True)

        total_loss, loss_mlm, loss_rec, loss_plain, loss_tra_mlm, correct_num, instance_num, ori_inputs, \
            correct_chars, wrong_chars, rank_correct_num, rank_instance_num, hit_correct, topk_ids, topk_scores = \
            model(images, mask_ori_images, input_ids, attn_masks, labels, plain_labels, pos_ids, type_ids, lengths,
                  mask_ids, mask_img_ids, mask_chs, mod)

        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_rec=loss_rec.item())
        metric_logger.update(loss_plain=loss_plain.item())
        metric_logger.update(loss_tra_mlm=loss_tra_mlm.item())
        metric_logger.update(correct_num=int(correct_num))
        metric_logger.update(instance_num=int(instance_num))
        update_map = {}
        for k in config['topk']:
            update_map[f'hit_correct_{k}'] = hit_correct[k]
            update_map[f'rank_correct_{k}'] = rank_correct_num[k]
            update_map[f'rank_instance_{k}'] = rank_instance_num[k]
        metric_logger.update(**update_map)

        if save_cases:
            for ch, idx in correct_chars:
                f_case.write(f'{tokenizer.convert_ids_to_tokens(ch)}\t{str(idx)}\n')
            f_case.write('\n')
            wrong_chars = [f'{tokenizer.convert_ids_to_tokens(ch)} {tokenizer.convert_ids_to_tokens(wch)} {str(idx)}'
                           for ch, wch, idx in wrong_chars]
            f_case.write('Wrong: ' + str(wrong_chars) + '\n\n')
            max_k = max(config['topk'])
            assert len(book_orders) == len(ori_inputs) == len(topk_ids) == len(topk_scores)
            for sent, book_order, topk_id, topk_score in zip(ori_inputs, book_orders, topk_ids, topk_scores):
                f_case.write(str(data_idx) + '\t' +
                             str(tokenizer.convert_ids_to_tokens(sent)) + '\t' + book_order + '\n')
                topk_chs, topk_pairs = tokenizer.convert_ids_to_tokens(topk_id), []
                for ch, score in zip(topk_chs, topk_score):
                    topk_pairs.append((ch, round(score, 4)))
                f_case.write(f'Top{max_k}: ' + str(topk_pairs) + '\n')
                data_idx += 1
            f_case.write('------------------------------\n\n')
        f_case.flush()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_default_metric()
    meters = metric_logger.meters
    res = {k: meter.metric_fmt.format(meter.default_metric) for k, meter in meters.items()}
    res['global_accuracy'] = round(100 * meters['correct_num'].total / meters['instance_num'].total, 2)
    hits = []
    for k in config['topk']:
        hit = round(100 * meters[f'hit_correct_{k}'].total / meters['instance_num'].total, 2)
        res[f'global_hit_{k}'] = hit
        res[f'global_rank_acc_{k}'] = round(
            100 * meters[f'rank_correct_{k}'].total / meters[f'rank_instance_{k}'].total, 2)
        hits.append(f'{hit:.2f}')
    print(f'topks: {config["topk"]} hits: {" / ".join(hits)}')

    if save_cases:
        f_case.close()

    return res


def test(args, config, model, data_loader, tokenizer=None):
    device = torch.device(args.device)

    # get all models
    if os.path.isdir(args.checkpoint):
        model_list = sorted([os.path.join(args.checkpoint, m)
                             for m in os.listdir(args.checkpoint) if m.endswith('.pth')])
    else:
        model_list = [args.checkpoint]

    for cp_path in model_list:
        # test every checkpoint
        checkpoint = torch.load(cp_path, map_location='cpu')
        state_dict, epoch = checkpoint['model'], checkpoint['epoch']
        inner_model = model.module if hasattr(model, 'module') else model
        if 'visual_encoder.pos_embed' in state_dict:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        msg = inner_model.load_state_dict(state_dict, strict=False)
        print('loading complete:', msg)
        print('load checkpoint from %s' % cp_path)

        start_time = time.time()

        test_stats = test_epoch(args, model, data_loader, epoch, device, config, tokenizer=tokenizer)
        if utils.is_main_process():
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

            print(log_stats)
            with open(os.path.join(config['output_path'], "log_test.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {} for epoch {}'.format(total_time_str, epoch))


def init_dataset(mode, config, distributed, tokenizer):
    # Dataset #
    print("Creating dataset")
    datasets = [create_dataset('finetune_single_mlm', mode, config, tokenizer)]

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [mode == 'train'], num_tasks, global_rank)
    else:
        samplers = [None]
    modality = config['modality'] if 'modality' in config else 'cross'
    assert modality in ['cross', 'text', 'image']

    def collate_fn(batch):
        return mlm_single_collate_fn(batch, tokenizer, modality)

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']],
                                num_workers=[4], is_trains=[mode == 'train'], collate_fns=[collate_fn])[0]
    return data_loader


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)

    train_loader, test_loader = None, None
    if args.mode in ['train', 'both']:
        train_loader = init_dataset('train', config, args.distributed, tokenizer)
    if args.mode in ['test', 'both']:
        test_loader = init_dataset('test', config, args.distributed, tokenizer)

    # Model #
    print("Creating model")
    model = name_to_model[config['model']](config=config, text_encoder=args.text_encoder,
                                           tokenizer=tokenizer, distributed=args.distributed)

    model = model.to(torch.device(args.device))

    if args.mode != 'test':
        train(args, config, model, train_loader, test_loader, tokenizer)
    else:
        test(args, config, model, test_loader, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Finetune_single_mlm.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', help='resume training', action='store_true')
    parser.add_argument('--text_encoder', default='')  # MODIFIED
    parser.add_argument('--text_tokenizer', default='../chinese-bert-wwm-ext')  # MODIFIED
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)  # MODIFIED
    parser.add_argument('--mode', choices=['train', 'test', 'both'], required=True)
    parser.add_argument('--save_all', default=True, type=bool)
    parser.add_argument('--load_cross', action='store_true')
    parser.add_argument('--text_checkpoint', default='', type=str)
    parser.add_argument('--image_checkpoint', default='', type=str)
    parser.add_argument('--test_files', help='multiple files seperated with \',\'', default='', type=str)
    parser.add_argument('--do_trans', choices=['', 'true', 'false'], default='', type=str)
    main_args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main_config = yaml.load(open(main_args.config, 'r'), Loader=yaml.Loader)
    if main_args.test_files != '':
        main_config['test_file'] = main_args.test_files.split(',')
    if main_args.do_trans != '':
        main_config['img_random_transform'] = True if main_args.do_trans == 'true' else False

    Path(main_config['output_path']).mkdir(parents=True, exist_ok=True)
    if main_args.mode in ['train', 'both']:
        os.makedirs(os.path.join(main_config['output_path'], 'logs_train'), exist_ok=True)
    if main_args.mode == 'both':
        os.makedirs(os.path.join(main_config['output_path'], 'logs_valid'), exist_ok=True)
    elif main_args.mode == 'test':
        os.makedirs(os.path.join(main_config['output_path'], 'logs_test'), exist_ok=True)

    yaml.dump(main_config, open(os.path.join(main_config['output_path'], 'config.yaml'), 'w'))

    main(main_args, main_config)
