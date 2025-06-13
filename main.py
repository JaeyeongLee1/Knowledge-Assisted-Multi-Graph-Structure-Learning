import random
import sys
import os
import argparse
import pandas as pd
import torch
import dill
import json

from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datetime import datetime

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc

from datasets.TimeDataset import TimeDataset
from models.Proposed import Proposed  #
from models.PF import PF
from models.SG import SG
from models.UN import UN

from train_kagsl2 import train  #
from test_kagsl import test
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt


class Main:
    def __init__(self, train_config_, env_config_, learned_time=None):
        self.train_config = train_config_
        self.env_config = env_config_
        self.datestr = learned_time

        dataset = self.env_config['dataset']
        if dataset == 'wadi':
            train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
            test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
        else:
            assert dataset == 'swat'
            train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
            test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

        assert (train_orig.columns == test_orig.columns).all()

        if 'attack' in train_orig.columns:
            train_orig = train_orig.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config_['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train_orig.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train_orig, feature_map, labels=0)
        test_dataset_indata = construct_data(test_orig, feature_map, labels=test_orig.attack.tolist())

        if self.train_config['scale_bool']:
            print('adjust mean scale')
            train_dataset_indata = torch.tensor(train_dataset_indata)
            test_dataset_indata = torch.tensor(test_dataset_indata)

            scale = train_dataset_indata[:-1].mean(-1).unsqueeze(-1)

            train_dataset_indata[:-1] = train_dataset_indata[:-1] - scale
            test_dataset_indata[:-1] = test_dataset_indata[:-1] - scale

            train_dataset_indata = train_dataset_indata.tolist()
            test_dataset_indata = test_dataset_indata.tolist()

        cfg = {
            'slide_win': train_config_['slide_win'],
            'slide_stride': train_config_['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg,
                                    device=env_config_['device'])
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg,
                                   device=env_config_['device'])

        train_dataloader, val_dataloader = self.get_loaders(train_dataset,
                                                            train_config_['seed'],
                                                            train_config_['batch'],
                                                            val_ratio=train_config_['val_ratio'],
                                                            slide_win=train_config_['slide_win'],
                                                            )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config_['batch'], shuffle=False, num_workers=0)

        process_dict = {
            'swat': [list(range(0, 5)), list(range(5, 16)), list(range(16, 25)), list(range(25, 34)),
                     list(range(34, 47)), list(range(47, 51))],
            'wadi': [list(range(0, 19)), list(range(19, 94)), list(range(94, 109)), list(range(109, 112))],
            'msl': [[0, 1, 2, 8, 9, 10, 25], [3], [4, 11, 20, 22], [5, 6, 14, 15, 19, 21],
                    [7, 16, 17, 26], [12, 13], [18, 23, 24]]
        }

        process_mat_swat = torch.zeros(6, 6)
        for i in range(5):
            process_mat_swat[i, i+1] = 1
        process_mat_swat[5, 0] = 1
        process_mat_swat[5, 2] = 1

        process_mat_wadi = torch.zeros(4, 4)
        process_mat_wadi[0, 1] = 1
        process_mat_wadi[1, 2] = 1
        process_mat_wadi[2, 0] = 1

        process_mat_dict = {
            'swat': process_mat_swat,
            'wadi': process_mat_wadi
        }

        process_info = process_dict[dataset]
        process_mat = process_mat_dict[dataset]
        mask1 = torch.zeros(len(feature_map), len(feature_map)).to(self.device)

        if dataset == 'wadi':  
            num_process = len(process_info) - 1
        else:
            assert dataset == 'swat'
            num_process = len(process_info)

        for s in range(num_process):
            for m in process_info[s]:
                for n in process_info[s]:
                    mask1[m, n] = 1

        mask2 = torch.zeros(len(feature_map), len(feature_map)).to(self.device)
        for s in range(len(process_info)):
            for r in range(len(process_info)):
                if process_mat[s, r] != 0.0:
                    for m in process_info[s]:
                        for n in process_info[r]:
                            mask2[m, n] = 1
                            mask2[n, m] = 1

        self.model = Proposed(fc_edge_index,
                              node_num=len(feature_map),
                              process_info=process_info,
                              mask1=mask1,
                              mask2=mask2,
                              emb_dim=train_config_['emb_dim'],
                              feature_dim=train_config_['feature_dim'],  # sensor_dim -> emb_dim and feature dim
                              slide_win=train_config_['slide_win'],
                              out_layer_num=train_config_['out_layer_num'],
                              out_layer_inter_dim=train_config_['out_layer_inter_dim'],
                              topk=train_config_['topk'],
                              alpha=train_config_['alpha'],
                              dropout=train_config_['dropout'],
                              out_dropout=train_config_['out_dropout'],
                              kernel_size=train_config_['kernel_size'],
                              out_mode=train_config_['out_mode'],
                              ).to(self.device)

    def run(self):
        parent_path, child_path = self.get_save_path()

        model_path = os.path.join(child_path, 'best_model.pt')
        fig_path = os.path.join(child_path, 'loss_plot.png')
        config_path = os.path.join(child_path, 'train_configs.txt')
        result_path = os.path.join(parent_path, f'metric_{self.datestr}.txt')

        with open(config_path, 'w') as f:
            json.dump(self.train_config, f, indent=2)

        train_losses, val_losses = train(model=self.model,
                                         save_path=model_path,
                                         config=self.train_config,
                                         train_dataloader=self.train_dataloader,
                                         val_dataloader=self.val_dataloader,
                                         )

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)

        plt.subplot(1, 2, 2)
        plt.plot(val_losses)

        plt.savefig(fig_path)
        plt.close()

        best_model = torch.load(model_path, pickle_module=dill).to(self.device)

        _, test_result = test(best_model, self.test_dataloader)
        _, val_result = test(best_model, self.val_dataloader)

        self.get_score(test_result, val_result, result_path)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1, slide_win=15):
        path_train_sub_indices = f'./data/{self.env_config["dataset"]}/' \
                                 f'train_sub_indices_{slide_win}_{val_ratio}_{seed}.pt'
        path_valid_sub_indices = f'./data/{self.env_config["dataset"]}/' \
                                 f'valid_sub_indices_{slide_win}_{val_ratio}_{seed}.pt'
        if os.path.exists(path_train_sub_indices) and os.path.exists(path_valid_sub_indices):
            train_sub_indices = torch.load(path_train_sub_indices)
            val_sub_indices = torch.load(path_valid_sub_indices)
        else:
            dataset_len = int(len(train_dataset))
            train_use_len = int(dataset_len * (1.0 - val_ratio))
            val_use_len = dataset_len - train_use_len
            #val_start_index = random.randrange(train_use_len)
            val_start_index = -val_use_len
            indices = torch.arange(dataset_len)

            #train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
            train_sub_indices = indices[:val_start_index]
            #val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
            val_sub_indices = indices[val_start_index:]

            torch.save(train_sub_indices, path_train_sub_indices)
            torch.save(val_sub_indices, path_valid_sub_indices)

        train_subset = Subset(train_dataset, train_sub_indices)
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True, drop_last=True)  # drop_last -> True

        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result, result_path):
        test_labels = test_result[2].tolist()

        if not os.path.exists(result_path):
            rs = open(result_path, 'w')
            head = 'settings, f1, pre, rec, TP, TN, FP, FN, threshold'

            rs.write(head)
            rs.close()

        
        b_num_list = [0, 1, 3, 5, 7, 9, 11, 13]
        epsilon_list = [1e-1, 1e-2, 1e-3, 1e-4]

        rs = open(result_path, 'a')
        for b_num in b_num_list:
            for epsilon in epsilon_list:
                settings = f"{self.env_config['report']}_b_num_{b_num}_eps_{epsilon}"
                print(settings)

                norm_test_scores, norm_val_scores = get_full_err_scores(test_result, val_result,
                                                                        epsilon=epsilon, b_num=b_num)

                if self.env_config['report'] == 'best':
                    metrics, threshold, _ = get_best_performance_data(norm_test_scores, test_labels, topk=1)
                else:
                    assert self.env_config['report'] == 'val'
                    metrics, threshold, _ = get_val_performance_data(norm_test_scores, norm_val_scores,
                                                                     test_labels, topk=1)

                result_summary = f'\n{settings},' + ','.join(f'{d: .4f}' for d in list(metrics + (threshold,)))
                rs.write(result_summary)

        rs.close()

    def get_save_path(self):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%b_%d_%H_%M_%S')

        if self.train_config['scale_bool']:
            scale_tag = 'scm'
        else:
            scale_tag = 'uscm'

        str_setting = f"sd_{self.train_config['seed']}_" \
                      f"lr_{self.train_config['lr']}_" \
                      f"sw_{self.train_config['slide_win']}_" \
                      f"vr_{self.train_config['val_ratio']}_" \
                      f"dem_{self.train_config['emb_dim']}_" \
                      f"dfe_{self.train_config['feature_dim']}_" \
                      f"ks_{self.train_config['kernel_size']}_" \
                      f"tk_{self.train_config['topk']}_" \
                      f"alp_{self.train_config['alpha']}_" \
                      f"nout_{self.train_config['out_layer_num']}_" \
                      f"dout_{self.train_config['out_layer_inter_dim']}_" \
                      f"dp_{self.train_config['dropout']}_" \
                      f"odp_{self.train_config['out_dropout']}_" \
                      f"om_{self.train_config['out_mode']}_" \
                      f"{scale_tag}"

        parent_path = f'./results/{dir_path}/{str_setting}'
        child_path = os.path.join(parent_path, self.datestr)

        Path(child_path).mkdir(parents=True, exist_ok=True)

        return parent_path, child_path


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument('-random_seed', help='random seed', type=int, default=0)

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=50)
    parser.add_argument('-lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('-early_stop_win', help='patience for earlystopping', type=int, default=5)
    parser.add_argument('-decay', help='decay', type=float, default=0.0)

    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')

    # emb_dim == feature_dim if out_mode == 4 or 5
    parser.add_argument('-emb_dim', help='dimension for embeddings', type=int, default=128)
    parser.add_argument('-feature_dim', help='dimension for features', type=int, default=128)
    parser.add_argument('-kernel_size', help='kernel_size', type=int, default=3)
    parser.add_argument('-topk', help='topk num', type=int, default=15)
    parser.add_argument('-alpha', help='alpha', type=float, default=0.3)
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=2)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=128)
    parser.add_argument('-dropout', help='dropout rate', type=float, default=0.0)
    parser.add_argument('-out_dropout', help='dropout rate for output layer', type=float, default=0.0)

    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='swat')
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')

    
    parser.add_argument('-scale_bool', action='store_true')
    parser.add_argument('-out_mode', type=int, default=4)

    
    '''sys.argv = [
        'main.py',
        '-random_seed', '0',

        '-batch', '64',
        '-epoch', '50',
        '-lr', '1e-3',
        '-early_stop_win', '5',
        '-decay', '0.0',

        '-slide_win', '15',
        '-slide_stride', '1',
        '-val_ratio', '0.1',
        # 'comment,

        '-emb_dim', '128',
        '-feature_dim', '128',
        '-kernel_size', '3',
        '-topk', '15',
        '-alpha', '0.3',
        '-out_layer_num', '2',
        '-out_layer_inter_dim', '128',
        '-dropout', '0.0',
        '-out_dropout', '0.0',
                        
        # '-save_path_pattern',
        '-dataset', 'swat',
        '-report', 'best',
        '-device', 'cuda',

        '-scale_bool',
        '-out_mode', '4',
    ]'''

    args = parser.parse_args()

    args.save_path_pattern = args.dataset
    args.comment = args.dataset

    """
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    """

    train_config = {
        'seed': args.random_seed,

        'batch': args.batch,
        'epoch': args.epoch,
        'lr': args.lr,
        'early_stop_win': args.early_stop_win,
        'decay': args.decay,

        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
        'val_ratio': args.val_ratio,
        'comment': args.comment,

        'emb_dim': args.emb_dim,
        'feature_dim': args.feature_dim,
        'kernel_size': args.kernel_size,
        'topk': args.topk,
        'alpha': args.alpha,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'dropout': args.dropout,
        'out_dropout': args.out_dropout,

        'scale_bool': args.scale_bool,
        'out_mode': args.out_mode,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
    }

    main = Main(train_config, env_config)
    main.run()
