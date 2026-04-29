#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from models.test import test_img, test_a3fl
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, get_model, ResNet18_cifar100, EfficientNetB0, ResNet18_FashionMNIST
from models.resnet20 import resnet20
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, sample_dirichlet_train_data, sample_iid_train_data
from utils.defense import fltrust, multi_krum, get_update, RLR, flame, get_update2, fld_distance, detection, detection1, parameters_dict_to_vector_flt, lbfgs_torch, layer_krum, flare
from models.Attacker import attacker
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
import yaml
import datetime
from utils.defense import *
from models.Update import *
from models.a3fl import A3FL


from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image

from utils.snowball import snowball

from utils.IndicatorServer import IndicatorServer, NoiseDataset, get_ood_dataloader



matplotlib.use('Agg')


def write_file(filename, accu_list, back_list, args, analyse=False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    # if args.defence == "krum":
    #     krum_file = filename + "_krum_dis"
    #     torch.save(args.krum_distance, krum_file)
    if args.defence == "flare":
        benign_file = filename + "_benign_dis.torch"
        malicious_file = filename + "_malicious_dis.torch"
        torch.save(args.flare_benign_list, benign_file)
        torch.save(args.flare_malicious_list, malicious_file)
        f.write('\n')
        f.write("avg_benign_list=")
        f.write(str(np.mean(args.flare_benign_list)))
        f.write('\n')
        f.write("avg_malicious_list=")
        f.write(str(np.mean(args.flare_malicious_list)))
    if analyse == True:
        need_length = len(accu_list) // 10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc), 2)
        average_back = round(np.mean(back), 2)
        best_back = round(max(back), 2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()


def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def compute_cosine_similarities(update1, update2):
    # 提取两个更新中的所有权重张量，并展平为向量
    vec1 = []
    vec2 = []

    # 假设两个更新有相同的键集合
    for key in update1.keys():
        # 跳过不需要的键
        if key.split('.')[-1] in ['num_batches_tracked', 'running_mean', 'running_var']:
            continue

        # 将张量展平并添加到向量列表中
        vec1.append(update1[key].flatten())
        vec2.append(update2[key].flatten())

    # 合并所有参数向量
    if vec1 and vec2:
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)

    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

    return similarity


if __name__ == '__main__':
    # parse args
    args = args_parser()
    if args.attack == 'lp_attack':
        args.attack = 'adaptive'  # adaptively control the number of attacking layers
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    test_mkdir('./' + args.save)
    print_exp_details(args)
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)

        # sample users
        if args.iid:
            dict_users = sample_iid_train_data(dataset_train, args.num_users)
        else:
            dict_users = sample_dirichlet_train_data(dataset_train, args.num_users)
            print('main_fed.py line 137 len(dict_users):', len(dict_users))

    elif args.dataset == 'cifar':
        if args.attack == 'adaptive':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10(
                '../data/', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10(
                '../data/', train=False, download=True, transform=trans_cifar)
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_train = datasets.CIFAR10(
                '../data/', train=True, download=True, transform=transform_train)
            dataset_test = datasets.CIFAR10(
                '../data/', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = sample_iid_train_data(dataset_train, args.num_users)
        else:
            dict_users = sample_dirichlet_train_data(dataset_train, args.num_users)

            print('main_fed.py line 137 len(dict_users):', len(dict_users))
    elif args.dataset == 'cifar100':
        if args.attack == 'adaptive':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR100(
                '../data/', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR100(
                '../data/', train=False, download=True, transform=trans_cifar)
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_train = datasets.CIFAR100(
                '../data/', train=True, download=True, transform=transform_train)
            dataset_test = datasets.CIFAR100(
                '../data/', train=False, download=True, transform=transform_test)

        if args.iid:
            dict_users = sample_iid_train_data(dataset_train, args.num_users)
        else:
            dict_users = sample_dirichlet_train_data(dataset_train, args.num_users)
            print('main_fed.py line 137 len(dict_users):', len(dict_users))
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist' or args.model == 'cnn' and args.dataset == 'fashion_mnist':
        net_glob = get_model('mnist').to(args.device)
    elif args.model == 'EfficientNet':
        net_glob = EfficientNetB0().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar100':
        net_glob = ResNet18_cifar100().to(args.device)
    elif args.model == "resnet" and args.dataset == 'fashion_mnist':
        net_glob = ResNet18_FashionMNIST().to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []

    args.flare_benign_list=[]
    args.flare_malicious_list=[]
    if args.defence == 'fld':
        old_update_list = []
        weight_record = []
        update_record = []
        args.frac = 1
        malicious_score = torch.zeros((1, 100))

    if args.defence == 'indicator':
        with open(f"./indicator_params.yaml", "r") as f:
            params_loaded = yaml.safe_load(f)
        ood_data = NoiseDataset(size=(1,28,28), num_samples=2000)
        ood_data = get_ood_dataloader(ood_data)
        server = IndicatorServer(params=params_loaded,
                                 open_set=ood_data, global_model=net_glob, args=args)

    if math.isclose(args.malicious, 0):
        backdoor_begin_acc = 100
    else:
        backdoor_begin_acc = args.attack_begin  # overtake backdoor_begin_acc then attack
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)  # get root dataset for FLTrust
    base_info = get_base_info(args)
    filename = './' + args.save + '/accuracy_file_{}.txt'.format(base_info)  # log hyperparameters

    if args.init != 'None':  # continue from checkpoint
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")

    val_acc_list = [0.0001]  # Acc list
    backdoor_acculist = [0]  # BSR list

    args.attack_layers = []  # keep LSA

    if args.attack == "dba":
        args.dba_sign = 0  # control the index of group to attack
    if args.log_distance == True:
        args.krum_distance = []
        args.krum_layer_distance = []
    malicious_list = []  # list of the index of malicious clients
    for i in range(int(args.num_users * args.malicious)):
        malicious_list.append(i)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    if args.defence == 'fedbap':
        defense_dataloader = []
        for i in range(args.num_users):
            ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users[i]), batch_size=args.local_bs, shuffle=True, drop_last=True)
            defense_dataloader.append(ldr_train)
        defense = FedBAP(args.num_users, 5, malicious_list, args.triggerY, args.triggerX, args.device, defense_dataloader)
    else:
        defense = None

    if args.attack == 'a3fl':
        a3fl_dl = DataLoader(DatasetSplit(dataset_train, dict_users[malicious_list[0]]), batch_size=args.local_bs,
                             shuffle=True, drop_last=True)
        a3fl_dl_test = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=True, drop_last=True)
        a3fl = A3FL(args.attack_label, args.poison_frac, args.triggerY, args.triggerX, 5, args.device)
    else:
        a3fl = None


    update_dict_last_round = {}
    for i in range(args.num_users):
        update_dict_last_round[i] = None
    similar_dict = {}
    for i in range(args.num_users):
        similar_dict[i] = []


    for iter in range(args.epochs):
        if defense is not None:
            defense.search_all_trigger(copy.deepcopy(net_glob).to(args.device), iter)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)  # number of clients in each round
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select the clients for a single round
        idxs_users = np.sort(idxs_users)

        if backdoor_begin_acc < val_acc_list[-1]:  # start attack only when Acc overtakes backdoor_begin_acc
            backdoor_begin_acc = 0
            attack_number = int(args.malicious * m)  # number of malicious clients in a single round
        else:
            attack_number = 0
            
        if args.scaling_attack_round != 1:
            # scaling attack begin 100-th round and perform each args.attack_round round
            if iter > 100 and iter%args.scaling_attack_round == 0:
                attack_number = attack_number
            else:
                attack_number = 0
        mal_weight=[]
        mal_loss=[]

        if args.attack == 'a3fl' and iter > a3fl.start:
            a3fl.search_trigger(net_glob, a3fl_dl)

        weight_aggregation = []

        if args.defence == 'indicator':
            server.global_model = net_glob
            server.pre_process(iter)


        for num_turn, idx in enumerate(idxs_users):
            if idx in malicious_list:

                mal_weight, loss, args.attack_layers = attacker(malicious_list, attack_number, args.attack,
                                                                dataset_train, dataset_test, dict_users, net_glob, args, a3fl, None, iter)
                w = mal_weight[0]

                if args.defence == 'snowball':
                    local = LocalUpdate(
                        args=args, dataset=dataset_train, idxs=dict_users[idx])
                    weight_aggregation.append(len(local.ldr_train))
            else:  # upload models for benign clients
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])

                w, loss = local.train(copy.deepcopy(net_glob).to(args.device), idx, defense, iter)

                if args.defence == 'snowball':
                    weight_aggregation.append(len(local.ldr_train))

            if update_dict_last_round[idx] is None:
                update_dict_last_round[idx] = get_update2(w, w_glob)
            else:
                similar_dict[idx].append(compute_cosine_similarities(get_update2(w, w_glob), update_dict_last_round[idx]))
                update_dict_last_round[idx] = get_update2(w, w_glob)


            if args.defence == 'fld' or args.defence == 'snowball':
                w_updates.append(get_update2(w, w_glob)) # ignore num_batches_tracked, running_mean, running_var
            else:
                w_updates.append(get_update(w, w_glob))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))



        if defense is not None:
            defense.modify_punish_factor(iter)
        if args.defence == 'avg' or args.defence == 'fedbap':  # no defence

            w_glob = FedAvg(w_locals)

        elif args.defence == 'indicator':
            if iter in server.watermarking_rounds:
                selected_client = server.indicator(w_locals)
                print(selected_client)
                w_glob = FedAvg([w_locals[x] for x in selected_client])
                #w_glob = FedAvg(w_locals)
            else:
                w_glob = FedAvg(w_locals)
        elif args.defence == 'snowball':
            selected_client = snowball(w_updates, dict_users, weight_aggregation,args)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'krum':  # single krum
            selected_client = multi_krum(w_updates, 1, args)
            w_glob = w_locals[selected_client[0]]
        elif args.defence == 'multikrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'RLR':
            w_glob = RLR(copy.deepcopy(net_glob), w_updates, args)
        elif args.defence == 'fltrust':
            local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
            fltrust_norm, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device))
            fltrust_norm = get_update(fltrust_norm, w_glob)
            w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)

        elif args.defence == 'fedtpd':
            fltrust_norm = get_central_param(w_updates, 1, args)
            w_glob = fedtpd(w_updates, fltrust_norm, w_glob, args)

        elif args.defence == 'dcagg':
            if iter > 1:
                result = classify_clients_smi(similar_dict)
                benign_client_smi = result['benign']
                features = result['features'].tolist()
                benign_client_hdb = classify_clients_hdb(w_updates)
                benign_client = list(set(benign_client_smi) & set(benign_client_hdb))
            else:
                benign_client = classify_clients_hdb(w_updates)
                features = [[1 for _ in range(len(w_updates))] for _ in range(len(w_updates))]

            fltrust_norm, selected_client = reference_gradient_extraction(w_updates, 0, args, benign_client)
            w_glob = dcagg(w_updates, fltrust_norm, w_glob, args, features, selected_client)


        elif args.defence == 'roseagg':
            w_updates = convert_w_updates_to_model_updates(w_updates)
            w_glob = roseagg(w_updates, w_glob, args)

        elif args.defence == 'flare':
            w_glob = flare(w_updates, w_locals, net_glob, central_dataset, dataset_test, w_glob, args)
        elif args.defence == 'flame':
            w_glob = flame(w_locals, w_updates, w_glob, args, debug=args.debug)
        elif args.defence == 'flip':
            w_glob = FedAvg(w_locals)
        elif args.defence == 'flip_multikrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'layer_krum':
            w_glob_update = layer_krum(w_updates, args.k, args, multi_k=True)
            for key, val in w_glob.items():
                w_glob[key] += w_glob_update[key]
            
        elif args.defence == 'fld':
            # ignore key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var'
            N = 5
            args.N = N
            weight = parameters_dict_to_vector_flt(w_glob)
            local_update_list = []
            for local in w_updates:
                local_update_list.append(-1*parameters_dict_to_vector_flt(local).cpu()) # change to 1 dimension
                
            if iter > N+1:
                args.hvp = lbfgs_torch(args, weight_record, update_record, weight - last_weight)
                hvp = args.hvp

                attack_number = int(args.malicious * m)
                distance = fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp)
                distance = distance.view(1,-1)
                malicious_score = torch.cat((malicious_score, distance), dim=0)
                if malicious_score.shape[0] > N+1:
                    if detection1(np.sum(malicious_score[-N:].numpy(), axis=0)):
                        
                        label = detection(np.sum(malicious_score[-N:].numpy(), axis=0), int(args.malicious * m))
                    else:
                        label = np.ones(100)
                    selected_client = []
                    for client in range(100):
                        if label[client] == 1:
                            selected_client.append(client)
                    new_w_glob = FedAvg([w_locals[client] for client in selected_client])
                else:
                    new_w_glob = FedAvg(w_locals)  # avg

            else:
                hvp = None
                new_w_glob = FedAvg(w_locals)  # avg

            update = get_update2(w_glob, new_w_glob)  # w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
            update = parameters_dict_to_vector_flt(update)
            if iter > 0:
                weight_record.append(weight.cpu() - last_weight.cpu())
                update_record.append(update.cpu() - last_update.cpu())
            if iter > N:
                del weight_record[0]
                del update_record[0]
            last_weight = weight
            last_update = update
            old_update_list = local_update_list
            w_glob = new_w_glob
        else:
            print("Wrong Defense Method")
            os._exit(0)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            if args.attack == 'a3fl':
                acc_test, _, _ = test_img(
                    net_glob, dataset_test, args, test_backdoor=True)
                back_acc = test_a3fl(net_glob, a3fl_dl_test, args, a3fl)
            else:
                acc_test, _, back_acc = test_img(
                    net_glob, dataset_test, args, test_backdoor=True)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("Backdoor accuracy: {:.2f}".format(back_acc))
            if args.model == 'lstm':
                val_acc_list.append(acc_test)
            else:
                val_acc_list.append(acc_test.item())

            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, args)

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)

    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label='main task(acc:' + str(best_acc) + '%)')
    plt.plot(backdoor_acculist, label='backdoor task(BBSR:' + str(bbsr) + '%, ABSR:' + str(absr) + '%)')
    plt.legend()
    title = base_info
    plt.title(title)
    plt.savefig('./' + args.save + '/' + title + '.pdf', format='pdf', bbox_inches='tight')

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    torch.save(net_glob.state_dict(), './' + args.save + '/model' + '.pth')