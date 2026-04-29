import torch
import torch.nn as nn

import numpy as np
import logging
import time
import copy
import math
import json
import random

from models.Nets import ResNet18, vgg19_bn, vgg19, get_model, vgg11, ResNet18_cifar100, EfficientNetB0, ResNet18_FashionMNIST, ResNet18_TinyImageNet

logger = logging.getLogger("logger")


class IndicatorServer:
    
    def __init__(self, params, open_set, global_model, args):
        super(IndicatorServer, self).__init__()
        self.params = params
        self.watermarking_rounds = [round for round in range(self.params["global_watermarking_start_round"],
                                                         self.params["global_watermarking_end_round"],
                                                         self.params["global_watermarking_round_interval"])] 

        ### add saved_models
        self.open_set=open_set


        self.poisoned_acc=[]
        self.clean_acc=[]
        self.no_detected_malicious = 0
        self.no_undetected_malicious = 0
        self.no_detected_benign = 0
        self.no_misclassified_benign = 0
        self.no_processed_malicious_clients = 0
        self.no_processed_benign_clients = 0
        self.VWM_detection_threshold = self.params["VWM_detection_threshold"]

        self.wm_mu = self.params["watermarking_mu"]

        self._create_additional_model(args)
        self._loss_function()

        self.after_wm_injection_bn_stats_dict = dict()
        self.global_model = global_model

    def _create_additional_model(self, args):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if args.model == 'VGG' and args.dataset == 'cifar':
            check_model = vgg19_bn().to(args.device)
        elif args.model == 'EfficientNet':
            check_model = EfficientNetB0().to(args.device)
        elif args.model == 'VGG11' and args.dataset == 'cifar':
            check_model = vgg11().to(args.device)
        elif args.model == "resnet" and args.dataset == 'cifar':
            check_model = ResNet18().to(args.device)
        elif args.model == "rlr_mnist" or args.model == "cnn":
            check_model = get_model('fmnist').to(args.device)
        elif args.model == "resnet" and args.dataset == 'cifar100':
            check_model = ResNet18_cifar100().to(args.device)
        elif args.dataset == 'imagenet':
            check_model = ResNet18_TinyImageNet().to(args.device)
        elif args.model == "resnet" and args.dataset == 'fashion_mnist':
            check_model = ResNet18_FashionMNIST().to(args.device)
        else:
            exit('Error: unrecognized model')
        
        self.check_model = check_model.cuda()
        return True



    def aggregation(self, weight_accumulator, aggregated_model_id):
        r"""
        aggregate all the updates model to generate a new global model
        """
        no_of_participants_this_round = sum(aggregated_model_id)
        for name, data in self.global_model.state_dict().items():
            # update_per_layer = weight_accumulator[name] * \
            #             (self.params["eta"] / self.params["no_of_participants_per_round"])
            update_per_layer = weight_accumulator[name] * \
                        (self.params["eta"] / no_of_participants_this_round)

            data = data.float()
            data.add_(update_per_layer)
        
        return True

    def indicator(self, local_model_state_dict):
        wm_data = self.open_set
        benign_client = []
        acc_list = []  # 记录每个客户端的 label_acc_w

        for ind, model_state_dict in enumerate(local_model_state_dict):
            self.check_model.load_state_dict(self.global_model.state_dict())

            for name, data in model_state_dict.items():
                if "num_batches_tracked" in name:
                    continue

                if "running" in name:
                    if self.params["replace_original_bn"]:
                        new_value = self.after_wm_injection_bn_stats_dict[name]
                    else:
                        continue
                else:
                    new_value = data.clone().detach()

                self.check_model.state_dict()[name].copy_(new_value)

            wm_copy_data = copy.deepcopy(wm_data)
            _, _, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(
                test_data=wm_copy_data, model=self.check_model
            )

            acc_list.append((ind, label_acc_w))

            if label_acc_w < self.VWM_detection_threshold:
                benign_client.append(ind)

        # 如果为空，就按label_acc_w降序取一个
        if len(benign_client) == 0:
            acc_list.sort(key=lambda x: x[1], reverse=True)  # 降序排序
            best_client = acc_list[0][0]
            benign_client.append(best_client)
        print(acc_list)
        return benign_client

    def _global_watermarking_test_sub(self, test_data, model=None):
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        dataset_size = 0
        correct = 0
        wm_label_correct = 0
        wm_label_sum = 0
        data_iterator = test_data

        wm_label_sum_list = [0 for i in range(self.params["class_num"])]
        wm_label_correct_list = [0 for i in range(self.params["class_num"])]
        wm_label_acc_list = [0 for i in range(self.params["class_num"])]
        wm_label_dict = dict()
        for i in range(self.params["class_num"]):
            wm_label_dict[i] = 0

        for batch_id, batch in enumerate(data_iterator):

            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += self.ceriterion(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            if batch_id==0 and model != None and self.params["show_train_log"]:
                logger.info(f"watermarking targets:{targets}")
                logger.info(f"watermarking pred :{pred}")
            
            for pred_item in pred:
                wm_label_dict[pred_item.item()]+=1

            # poisoned_label = self.params["poison_label_swap"]
            for target_label in range(self.params["class_num"]):
                wm_label_targets = torch.ones_like(targets) * target_label
                wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))

                wm_label_sum_list[target_label] += wm_label_index.cpu().sum().item()
                wm_label_correct_list[target_label] += pred.eq(targets.data.view_as(pred))[wm_label_index.bool()].cpu().sum().item() 

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            dataset_size += len(targets)
            
        watermark_acc = 100.0 *(float(correct) / float(dataset_size))
        for i in range(self.params["class_num"]):
            wm_label_dict[i] = round(wm_label_dict[i]/dataset_size,2)
        for target_label in range(self.params["class_num"]):
            wm_label_acc_list[target_label] = round(100.0 * (float(wm_label_correct_list[target_label]) / float(wm_label_sum_list[target_label])), 2)

        # wm_label_acc = 100.0 * (float(wm_label_correct) / float(wm_label_sum))
        wm_label_acc = max(wm_label_acc_list)
        wm_index_label = wm_label_acc_list.index(wm_label_acc)
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, watermark_acc, wm_label_acc, wm_index_label, wm_label_acc_list, wm_label_dict)


    def ceriterion_build(self, input, target, reduction=None):
        loss = nn.functional.cross_entropy(input, target, reduction=reduction)
        return loss

    def _loss_function(self):
        self.ceriterion = self.ceriterion_build
        return True

    def _optimizer(self, round, model):
        lr = self.params["global_lr"]
        momentum = self.params["global_momentum"] 
        weight_decay = self.params["global_weight_decay"] 

        logger.info(f"indicator lr:{lr}")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        return True

    def _scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['global_milestones'],
                                                 gamma=self.params['global_lr_gamma'])
        return True

    def _projection(self, target_params_variables):
        model_norm = self._model_dist_norm(self.global_model, target_params_variables)
        if model_norm > self.params["global_projection_norm"] and self.params["global_is_projection_grad"]:
            norm_scale = self.params["global_projection_norm"] / model_norm
            for name, param in self.global_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)
        return True

    def _model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    def _model_dist_norm_var(self, model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def _global_watermark_injection(self, watermark_data, target_params_variables, round=None, model=None):

        if model==None:
            model = self.global_model
        model.train()

        total_loss = 0
        self._loss_function()
        self._optimizer(round, model)
        self._scheduler()


        retrain_no_times = self.params["global_retrain_no_times"]
        
        for internal_round in range(retrain_no_times):

            data_iterator = copy.deepcopy(watermark_data)

            for batch_id, watermark_batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                wm_data, wm_targets = watermark_batch                
                wm_data = wm_data.cuda().detach().requires_grad_(False)
                wm_targets = wm_targets.cuda().detach().requires_grad_(False)

                data = wm_data
                targets = wm_targets

                output = model(data) 
                pred = output.data.max(1)[1]

                class_loss = nn.functional.cross_entropy(output, targets)
                distance_loss = self._model_dist_norm_var(model, target_params_variables)
                loss = class_loss + (self.wm_mu/2) * distance_loss 

                loss.backward()
                self.optimizer.step()
                
                self._projection(target_params_variables)
                total_loss += loss.data

            self.scheduler.step()

        return True


    def pre_process(self, round):

        ### Initialize to calculate the distance between updates and global model
        if round in self.watermarking_rounds:
            target_params_variables = dict()
            for name, param in self.global_model.state_dict().items():
                target_params_variables[name] = param.clone()

            before_wm_injection_bn_stats_dict = dict()
            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    before_wm_injection_bn_stats_dict[key] = value.clone().detach()

            wm_data = copy.deepcopy(self.open_set)
            self._global_watermark_injection(watermark_data=wm_data,
                            target_params_variables=target_params_variables,
                            model=self.global_model,
                            round=round
                            )


            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.after_wm_injection_bn_stats_dict[key] = value.clone().detach()

            self.check_model.load_state_dict(self.global_model.state_dict())

            for key, value in self.check_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.check_model.state_dict()[key].\
                        copy_(before_wm_injection_bn_stats_dict[key])
                    if self.params["replace_original_bn"]:
                        self.global_model.state_dict()[key].\
                            copy_(before_wm_injection_bn_stats_dict[key])

        return True

    def post_process(self):
        return True


class NoiseDataset(torch.utils.data.Dataset):

    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        noise = noise.cuda()
        return noise, torch.tensor(0)


def get_ood_dataloader(ood_dataset):
    r'''
    sample limited ood data as open set noise
    '''
    indices = random.sample(range(len(ood_dataset)), 2000)

    ood_dataloader =  torch.utils.data.DataLoader(ood_dataset,
                                       batch_size=64,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                       drop_last=True)
    ood_datalist = list(ood_dataloader)
    ood_datalist_shape = 2000//64 * 64
    assigned_labels = np.array([i for i in range(10)] * \
        (ood_datalist_shape//10) + [i for i in range(ood_datalist_shape%10)])
    np.random.shuffle(assigned_labels)
    assigned_labels = assigned_labels.reshape(2000//64, 64)
    for batch_id, batch in enumerate(ood_datalist):
        data, targets = batch

        for ind in range(len(targets)):
            targets[ind] = assigned_labels[batch_id][ind]
    ood_dataloader=iter(ood_datalist)
    return ood_dataloader