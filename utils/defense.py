# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from models.Update import LocalUpdate
import heapq
import math
import torch.nn.functional as F
from tqdm import *

class FedBAP:
    def __init__(self, num_clients, trigger_size, malicious_clients, triggerY, triggerX, device, dataloader):

        self.num_clients = num_clients
        self.triggers = []
        self.if_searched = []
        self.device = device
        self.malicious_clients = malicious_clients
        self.init_triggers(num_clients)
        self.punish_factor = 1
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        self.start_epoch = 100
        self.epoch = 10
        self.delta = 1
        self.punish_loss = {}
        self.dataloader = dataloader

        self.alpha_list = []
        self.window_size = 5

        self.num_classes = 10
        self.trigger_steps = 100
        self.epsilon = 1e-7
        self.attack_succ_threshold = 0.6

    def init_triggers(self, num_clients):
        for i in range(num_clients):
            self.triggers.append(torch.randn((1, 3, 32, 32), requires_grad=False, device=self.device))
            self.if_searched.append(False)
        self.mask = torch.zeros_like(self.triggers[0])

        self.mask = self.mask.to(self.device)

    def poison(self, client_id, inputs):
        return self.triggers[client_id] * self.mask + (1 - self.mask) * inputs

    def punish_loss(self, client_id, model, inputs, labels, epoch):
        loss = 0
        if client_id not in self.malicious_clients and epoch >= self.start_epoch:
            outputs = model(self.poison(client_id, inputs))
            loss = self.loss_fn(outputs, labels) * self.punish_factor
        return loss

    def train(self, dataloader, model, optimizer, participant_id, epoch):
        if epoch >= self.start_epoch and participant_id not in self.malicious_clients:
            epoch_loss = []
            for e in range(self.epoch):
                batch_loss = []
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(self.poison(participant_id, inputs))
                    loss = self.loss_fn(outputs, labels) * self.punish_factor
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if epoch not in self.punish_loss.keys():
                self.punish_loss[epoch] = []
            self.punish_loss[epoch].append(sum(epoch_loss) / len(epoch_loss))
            return sum(epoch_loss) / len(epoch_loss)
        else:
            return -1

    def modify_punish_factor(self, epoch):
        if epoch < self.start_epoch:
            return
        punish_loss = sum(self.punish_loss[epoch]) / len(self.punish_loss[epoch])
        if epoch == self.start_epoch:
            last_punish_loss = punish_loss
            beta = 1
        else:
            last_punish_loss = sum(self.punish_loss[epoch - 1]) / len(self.punish_loss[epoch - 1])
            if epoch - self.start_epoch < self.window_size:
                beta = sum(self.alpha_list) / (epoch - self.start_epoch)
            else:
                beta = sum(self.alpha_list[-self.window_size:]) / self.window_size
        alpha = punish_loss / last_punish_loss

        self.punish_factor += self.function(alpha) * beta

        self.alpha_list.append(alpha)


    def function(self, x):
        if x > 1:
            return self.delta / math.sqrt(x)
        else:
            return self.delta / x**2

    def trigger_loss(self, model, with_trigger, without_trigger):
        batch_size = with_trigger.shape[0]
        # 字典用于存储两次前向传播中倒数第二层的输出
        outputs = []

        # 定义一个钩子函数，捕获倒数第二层的输出
        def hook_fn(module, input, output):
            # 根据调用顺序存储不同输入的输出
            outputs.append(output.view(batch_size, -1))

        # 获取模型的所有层，并找到倒数第二层
        layers = list(model.children())
        layer_to_hook = layers[-1]

        # 在倒数第二层上注册前向钩子
        hook = layer_to_hook.register_forward_hook(hook_fn)

        # 计算第一个输入的输出
        output1 = model(with_trigger)

        # 计算第二个输入的输出
        output2 = model(without_trigger)

        # 取消钩子的注册
        hook.remove()

        cosine_sim = F.cosine_similarity(outputs[0], outputs[1], dim=1).mean()

        return cosine_sim


    def search_all_trigger(self, model, epoch):
        self.search_mask(model, epoch)
        if epoch == self.start_epoch:
            alpha = 10
            K = 100
            model.eval()

            for param in model.parameters():
                param.requires_grad = False

            for id in tqdm(range(self.num_clients)):
                if id not in self.malicious_clients:
                    t = self.triggers[id].clone()
                    m = self.mask.clone()

                    for iter in range(K):
                        loss_epoch = []
                        for inputs, _ in self.dataloader[id]:
                            t.requires_grad_()
                            inputs = inputs.to(self.device)
                            inputs_with_trigger = t * m + (1 - m) * inputs
                            loss = self.trigger_loss(model, inputs, inputs_with_trigger)
                            loss_epoch.append(loss.item())

                            loss.backward()
                            new_t = t - alpha * t.grad
                            t = new_t.detach_()
                            # t = torch.clamp(t, min=-2, max=2)

                    t = t.detach()
                    self.triggers[id] = t

            for param in model.parameters():
                param.requires_grad = True

            model.train()


    def search_mask(self, model, epoch):
        if epoch == self.start_epoch:
            all_mask_list = []
            learning_rate = 0.1
            cost = 1e-3
            model.eval()
            for id in tqdm(range(self.num_clients)):
                if id not in self.malicious_clients:
                    y_target = 0
                    best_target = 0
                    best_acc = 0
                    reg_best = float("inf")

                    criterion = torch.nn.CrossEntropyLoss(reduction="none")
                    while y_target < self.num_classes:

                        init_mask = np.random.random((1, 32, 32)) * 0.1

                        init_pattern = np.random.random((1, 3, 32, 32))

                        init_mask = np.clip(init_mask, 0.0, 1.0)
                        init_mask = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                        init_pattern = np.clip(init_pattern, 0.0, 1.0)
                        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

                        mask_tensor = torch.Tensor(init_mask).to(self.device)
                        pattern_tensor = torch.Tensor(init_pattern).to(self.device)
                        mask_tensor.requires_grad = True
                        pattern_tensor.requires_grad = True


                        optimizer = torch.optim.Adam(
                            [mask_tensor, pattern_tensor], lr=learning_rate, betas=(0.5, 0.9)
                        )
                        for iter in range(self.trigger_steps):
                            loss_ce_list = []
                            loss_reg_list = []
                            loss_list = []
                            acc_list = []
                            for inputs, _ in self.dataloader[id]:
                                inputs = inputs.to(self.device)
                                mask = (
                                        torch.tanh(mask_tensor) / (2 - self.epsilon) + 0.5
                                )
                                pattern = (
                                        torch.tanh(pattern_tensor) / (2 - self.epsilon) + 0.5
                                )
                                inputs = pattern * mask + (1 - mask) * inputs
                                labels = torch.full((inputs.shape[0],), y_target).to(self.device)
                                output = model(inputs)
                                optimizer.zero_grad()
                                loss_ce = criterion(output, labels)
                                loss_reg = torch.sum(torch.abs(mask)) / 3
                                loss = loss_ce.mean() + loss_reg * cost
                                loss.backward()
                                optimizer.step()

                                pred = output.argmax(dim=1, keepdim=True)
                                acc = pred.eq(labels.view_as(pred)).sum().item() / inputs.shape[0]
                                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                                loss_list.append(loss.detach().cpu().numpy())
                                acc_list.append(acc)
                            avg_loss_ce = np.mean(loss_ce_list)
                            avg_loss_reg = np.mean(loss_reg_list)
                            avg_loss = np.mean(loss_list)
                            avg_acc = np.mean(acc_list)

                            if avg_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                                mask_best = mask
                                pattern_best = pattern
                                reg_best = avg_loss_reg
                                best_target = y_target
                                best_acc = avg_acc
                                epsilon = 0.01
                                init_mask = mask_best + torch.distributions.Uniform(
                                    low=-epsilon, high=epsilon
                                ).sample(mask_tensor.shape).to(self.device)
                                init_mask = torch.clip(init_mask, 0.0, 1.0)
                                init_mask = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                                init_pattern = pattern_best + torch.distributions.Uniform(
                                    low=-epsilon, high=epsilon
                                ).sample(pattern_tensor.shape).to(self.device)
                                init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                                init_pattern = torch.arctanh(
                                    (init_pattern - 0.5) * (2 - self.epsilon)
                                )

                                with torch.no_grad():
                                    mask_tensor.copy_(init_mask)
                                    pattern_tensor.copy_(init_pattern)
                        # print(y_target, avg_acc, avg_loss_reg)
                        y_target += 1
                    all_mask_list.append(mask_best)
                    print(best_target, best_acc)

            model.train()


            self.mask = torch.mean(torch.stack(all_mask_list), dim=0).detach()
            self.mask = (self.mask > 0.5).float()



def cos(a, b):
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


import torch

def kernel_function(x, y):
    sigma = 1.0
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y):
    # Compute the MMD between two tensors x and y
    # x and y should have the same number of samples
    m = x.size(0)
    n = y.size(0)
    # Compute the kernel matrices for x and y
    xx_kernel = torch.zeros((m, m))
    yy_kernel = torch.zeros((n, n))
    xy_kernel = torch.zeros((m, n))
    for i in range(m):
        for j in range(i, m):
            xx_kernel[i, j] = xx_kernel[j, i] = kernel_function(x[i], x[j])

    for i in range(n):
        for j in range(i, n):
            yy_kernel[i, j] = yy_kernel[j, i] = kernel_function(y[i], y[j])

    for i in range(m):
        for j in range(n):
            xy_kernel[i, j] = kernel_function(x[i], y[j])
    # Compute the MMD statistic
    mmd = (torch.sum(xx_kernel) / (m * (m - 1))) + (torch.sum(yy_kernel) / (n * (n - 1))) - (2 * torch.sum(xy_kernel) / (m * n))
    return mmd


def flare(w_updates, w_locals, net, central_dataset, dataset_test, global_parameters, args):
    w_feature=[]
    temp_model = copy.deepcopy(net)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    
    for client in w_locals:
        net.load_state_dict(client)
        local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
        feature = local.get_PLR(
            net=copy.deepcopy(net).to(args.device))
        w_feature.append(feature)
    distance_list=[[] for i in range(len(w_updates))]
    # distance_list=[list(len(w_updates)) for i in range(len(w_updates))]
    for i in range(len(w_updates)):
        for j in range(i+1, len(w_updates)):
            score = compute_mmd(w_feature[i], w_feature[j])
            distance_list[i].append(score.item())
            distance_list[j].append(score.item())
    print('defense line121 distance_list', distance_list)
    vote_counter=[0 for i in range(len(w_updates))]
    k = round(len(w_updates)*0.5)
    for i in range(len(w_updates)):
        IDs = np.argsort(distance_list[i])
        for j in range(len(IDs)):
            # client_id is the index of client i-th client voting for
            # distance_list[] only records score with other clients without itself
            # so distance_list[i][i] should be itself
            # client_id = j + 1 after j >= i
            if IDs[j] >= i:
                client_id = IDs[j] + 1 
            else:
                client_id = IDs[j]
            vote_counter[client_id] += 1
            if j + 1 >= k:  # first 𝑘 elements in 𝐼 𝐷𝑠 and vote for it
                break

    trust_score = [x/sum(vote_counter) for x in vote_counter]
    # print('defense line188 len trust_score', trust_score)
    
    w_avg = copy.deepcopy(global_parameters)
    for k in w_avg.keys():
        for i in range(0, len(w_updates)):
            try:
                w_avg[k] += w_updates[i][k] * trust_score[i]
            except:
                print("Fed.py line17 type_as", 'w_updates[i][k].type():', w_updates[i][k].type(), k)
                w_updates[i][k] = w_updates[i][k].type_as(w_avg[k]).long()
                w_avg[k] = w_avg[k].long() + w_updates[i][k] * trust_score[i]
    return w_avg


def log_layer_wise_distance(updates):
    # {layer_name, [layer_distance1, layer_distance12...]}
    layer_distance = {}
    for layer, val in updates[0].items():
        if 'num_batches_tracked' in layer:
            continue
        # for each layer calculate distance among models
        for model in updates:
            temp_layer_dis = 0
            for model2 in updates:
                temp_norm = torch.norm((model[layer] - model2[layer]))
                temp_layer_dis += temp_norm
            if layer not in layer_distance.keys():
                layer_distance[layer] = []
            layer_distance[layer].append(temp_layer_dis.item())
    return layer_distance
    
        
def layer_krum(gradients, n_attackers, args, multi_k=False):
    new_global = {}
    for layer in gradients[0].keys():
        if layer.split('.')[-1] == 'num_batches_tracked' or layer.split('.')[-1] == 'running_mean' or layer.split('.')[-1] == 'running_var':
            new_global[layer] = gradients[-1][layer]
        else:
            layer_gradients = [x[layer] for x in gradients]
            new_global[layer] = layer_multi_krum(layer_gradients, n_attackers, args, multi_k)
    return new_global

def layer_flatten_grads(gradients):
    flat_epochs = []
    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        flat_epochs.append(gradients[n_user].cpu().numpy().flatten().tolist())
    flat_epochs = np.array(flat_epochs)
    return flat_epochs

def layer_multi_krum(layer_gradients, n_attackers, args, multi_k=False):
    grads = layer_flatten_grads(layer_gradients)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    score_record = None
    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        if args.log_distance == True and score_record == None:
            print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
            # print('defense.py line149 (layer_distance_dict):', layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    agg_layer = 0
    for selected_layer in candidate_indices:
        agg_layer += layer_gradients[selected_layer]
    agg_layer /= len(candidate_indices)
    return agg_layer

def multi_krum(gradients, n_attackers, args, multi_k=False):
    grads = flatten_grads(gradients)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    
    score_record = None

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        
        if args.log_distance == True and score_record == None:
            print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # num_clients = max(int(args.frac * args.num_users), 1)
    # num_malicious_clients = int(args.malicious * num_clients)
    # args.turn+=1
    # for selected_client in candidate_indices:
    #     if selected_client < num_malicious_clients:
    #         args.wrong_mal += 1
    #
    # for i in range(len(scores)):
    #     if i < num_malicious_clients:
    #         args.mal_score += scores[i]
    #     else:
    #         args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs



def get_update_filp(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = torch.div(update[key] - model[key], -1)
    return update2


def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        model_tensor = model[key].to(var.device)  # 确保 model[key] 和 update[key] 在同一设备
        update2[key] = var - model_tensor
    return update2

def get_update2(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        update2[key] = update[key] - model[key]
    return update2


def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))
        
    
    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)
    
    distance = torch.norm((old_update_list - local_update_list), dim=1)
    print('defense line219 distance(old_update_list - local_update_list):',distance)

    distance = torch.norm((pred_update - local_update_list), dim=1)
    distance = distance / torch.sum(distance)
    return distance

def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
    return label_pred

def detection1(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    select_k = 2  # default detect attacks
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1

def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    print('defense.py line 430 lr_vector == -1', lr_vector[lr_vector==-1].shape[0]/lr_vector.view(-1).shape[0])
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def weighted_average_vectors(vectors):
    num_vectors = len(vectors)
    if num_vectors == 1:
        return vectors[0]

    weights = []
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)

    for i in range(num_vectors):
        similarity_sum = 0
        for j in range(num_vectors):
            if i != j:
                similarity = cosine_similarity(vectors[i], vectors[j])
                similarity_sum += similarity
        weights.append(similarity_sum)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()

    weighted_sum = torch.zeros_like(vectors[0])
    for i in range(num_vectors):
        weighted_sum += weights[i] * vectors[i]

    return weighted_sum

def trimmed_mean(tensor_list, beta):
    # 将张量列表堆叠成一个新的张量，新增维度在第0维
    stacked_tensors = torch.stack(tensor_list)
    num_tensors = stacked_tensors.size(0)

    # 若能截断，执行截断操作；否则直接计算均值
    if 2 * beta < num_tensors:
        # 对第0维排序
        sorted_values, _ = torch.sort(stacked_tensors, dim=0)
        # 去除 beta 个最大值和最小值
        trimmed_values = sorted_values[beta:-beta]
        # 计算截断均值
        result = torch.mean(trimmed_values, dim=0)
    else:
        result = torch.mean(stacked_tensors, dim=0)

    return result

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics.pairwise import cosine_distances

def cluster_fun(coses, k, clustering_method='KMeans'):
    if clustering_method == 'Agglomerative':
        clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)
    elif clustering_method == 'KMeans':
        clustering = KMeans(n_clusters=k).fit(coses)
    elif clustering_method == 'Spectral':
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(coses)
    elif clustering_method == 'hdbscan':
        clustering = hdbscan.HDBSCAN(min_cluster_size=k, metric='precomputed').fit(np.array(coses, dtype=np.float64))
    else:
        raise NotImplementedError
    return clustering


def get_optimal_k_for_clustering(grads, clustering_method='KMeans'):
    coses = []
    nets = grads
    coses = cosine_distances(nets, nets)
    coses = np.array(coses)
    np.fill_diagonal(coses, 0)
    # logger.info(f'coses: {coses}')

    sil = []
    minval = 2

    #for k in range(minval, min(len(nets), 15)):
    for k in range(minval, len(nets)):
        clustering = cluster_fun(coses, k, clustering_method)
        labels = clustering.labels_
        # print(labels)
        sil.append(silhouette_score(coses, labels, metric='precomputed'))
    # logger.info(f'Silhouette scores: {sil}')
    return sil.index(max(sil)) + minval, coses



def select_representative_parameter(clus_represent_parm):
    min_distance_sum = float('inf')
    representative_parameter = None
    for key1, param1 in clus_represent_parm.items():
        distance_sum = 0
        for key2, param2 in clus_represent_parm.items():
            if key1 != key2:
                # 计算欧几里得距离
                distance = torch.norm(param1 - param2)
                distance_sum += distance
        if distance_sum < min_distance_sum:
            min_distance_sum = distance_sum
            representative_parameter = param1
    return representative_parameter


from scipy.stats import pearsonr


def compute_correlation_matrix(curves):
    """计算曲线间的皮尔逊相关系数矩阵，作为走势相似度度量"""
    n = len(curves)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # 计算两条曲线的皮尔逊相关系数，衡量走势相似性
            corr, _ = pearsonr(curves[i], curves[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # 对称矩阵

    return corr_matrix


def classify_clients_smi(curves_dict):

    curves = list(curves_dict.values())

    curves = np.array(curves)

    # 1. 相似度矩阵
    corr_matrix = compute_correlation_matrix(curves)

    # 2. 每个客户端用“和其他客户端的相关系数”作为特征
    features = corr_matrix

    # 3. 聚成两类
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # clusters = kmeans.fit_predict(features)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=len(features) // 2 + 1, min_samples=1, allow_single_cluster=True).fit(features)
    clusters = clusterer.labels_

    # 5. 判断多数簇为良性，少数簇为恶意
    unique, counts = np.unique(clusters, return_counts=True)
    benign_cluster = unique[np.argmax(counts)]
    benign_indices = [i for i, c in enumerate(clusters) if c == benign_cluster]
    malicious_indices = [i for i, c in enumerate(clusters) if c != benign_cluster]

    print(f"benign: {len(benign_indices)}, malicious: {len(malicious_indices)}")
    print(benign_indices)

    return {
        "benign": benign_indices,
        "malicious": malicious_indices,
        "features": features
    }

def classify_clients_hdb(update_params):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    local_model_vector = []
    for param in update_params:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)

    num_clients = len(update_params)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
        cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(update_params)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max() + 1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)

    return benign_client


def are_dicts_equal(dict1, dict2, atol=1e-6):
    """
    判断两个包含张量的字典是否相等

    参数:
        dict1, dict2: 要比较的两个字典
        atol: 浮点数比较的容差（处理数值精度问题）

    返回:
        如果字典结构和内容都相等则返回True，否则返回False
    """
    # 先检查键是否完全相同
    if dict1.keys() != dict2.keys():
        return False

    # 逐个比较字典中的张量值
    for key in dict1:
        # 检查对应的值是否都是张量
        if not (isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor)):
            return False

        # 比较张量（根据数据类型选择合适的方法）
        if not torch.allclose(dict1[key], dict2[key], atol=atol):
            return False

    return True

def reference_gradient_extraction(update_params, n_attackers, args, benign_client):

    w_updates_copy = [update_params[i] for i in benign_client]
    selected_client = multi_krum(w_updates_copy, n_attackers, args)

    # 找到目标张量在列表中的索引
    target_tensor = w_updates_copy[selected_client[0]]
    index = None
    for i, current_dict in enumerate(update_params):
        if are_dicts_equal(current_dict, target_tensor):
            index = i
            break
    print(index)
    return w_updates_copy[selected_client[0]], index



def sigmoid_enhance_normalize(lst, alpha=3):
    """
    反sigmoid增强：通过调整斜率放大差异
    alpha: 斜率参数（值越大，差异越显著）
    """

    if not lst:
        return []

    # 先Min-Max归一化到[0,1]，再映射到[-1,1]范围
    min_val = min(lst)
    max_val = max(lst)
    if max_val == min_val:
        return [1 for _ in lst]
    min_max_norm = [(x - min_val) / (max_val - min_val) for x in lst]
    scaled = [2 * x - 1 for x in min_max_norm]  # 映射到[-1,1]

    # 用tanh函数增强：接近1的值更大，接近-1的值更小
    enhanced = [math.tanh(alpha * x) for x in scaled]

    # 映射回[0,1]
    final_norm = [(x + 1) / 2 for x in enhanced]
    return final_norm


def dcagg(params, central_param, global_parameters, args, features, selected_client):
    feature = sigmoid_enhance_normalize(features[selected_client])


    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

    feature_matrix = []
    for param in params:
        feature_matrix.append(parameters_dict_to_vector_flt(param).to('cpu'))

    k, coses = get_optimal_k_for_clustering(feature_matrix)
    clustering = cluster_fun(coses, k)
    cluster_labels = clustering.labels_


    print(cluster_labels)

    clus_trust = {}
    clus_normalizing = {}
    clus_represent_parm = {}


    for i, label in enumerate(cluster_labels):
        if label not in clus_represent_parm:
            clus_represent_parm[label] = []
        clus_represent_parm[label].append(feature_matrix[i])


    for clus, vectors in clus_represent_parm.items():
        # 均值
        # stacked = torch.stack(vectors, dim=0)
        # clus_represent_parm[clus] = torch.mean(stacked, dim=0)

        # # 中值
        stacked = torch.stack(vectors, dim=0)
        clus_represent_parm[clus] = torch.median(stacked, dim=0).values

        # # 截断中值
        # clus_represent_parm[clus] = trimmed_mean(vectors, 1)

        # roseagg
        # clus_represent_parm[clus] = weighted_average_vectors(vectors)

    central_vector = parameters_dict_to_vector_flt(central_param)
    central_vector = central_vector.to('cpu')
    central_norm = torch.norm(central_vector)

    for clus, vector in clus_represent_parm.items():
        clus_trust[clus] = cos(vector, central_vector)
        clus_normalizing[clus] = central_norm / torch.norm(vector)

    sum_parameters = None
    FLTrustTotalScore = 0
    score_list = []
    for i, local_parameters in enumerate(feature_matrix):
        client_cos = max(clus_trust[cluster_labels[i]].item(), 0)
        client_clipped_value = clus_normalizing[cluster_labels[i]]
        score_list.append(client_cos * client_clipped_value * feature[i])
        FLTrustTotalScore += client_cos * client_clipped_value * feature[i]
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = client_cos * \
                                      client_clipped_value * var.clone() * feature[i]
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * params[i][
                    var] * feature[i]

    if FLTrustTotalScore == 0:
        # print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    # print(score_list)
    return global_parameters




def get_central_param(update_params, n_attackers, args):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    local_model_vector = []
    for param in update_params:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)

    num_clients = max(int(args.frac * args.num_users), 1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
        cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(update_params)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max() + 1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)

    w_updates = [update_params[i] for i in benign_client]
    selected_client = multi_krum(w_updates, n_attackers, args)
    print(selected_client[0])
    return w_updates[selected_client[0]]


def fedtpd(params, central_param, global_parameters, args):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

    feature_matrix = []
    for param in params:
        feature_matrix.append(parameters_dict_to_vector_flt(param).to('cpu'))

    k, coses = get_optimal_k_for_clustering(feature_matrix)
    clustering = cluster_fun(coses, k)
    cluster_labels = clustering.labels_

    print(cluster_labels)

    clus_trust = {}
    clus_normalizing = {}
    clus_represent_parm = {}


    for i, label in enumerate(cluster_labels):
        if label not in clus_represent_parm:
            clus_represent_parm[label] = []
        clus_represent_parm[label].append(feature_matrix[i])


    for clus, vectors in clus_represent_parm.items():
        # 均值
        # stacked = torch.stack(vectors, dim=0)
        # clus_represent_parm[clus] = torch.mean(stacked, dim=0)

        # # 中值
        stacked = torch.stack(vectors, dim=0)
        clus_represent_parm[clus] = torch.median(stacked, dim=0).values

        # # 截断中值
        # clus_represent_parm[clus] = trimmed_mean(vectors, 1)

        # roseagg
        # clus_represent_parm[clus] = weighted_average_vectors(vectors)

    central_vector = parameters_dict_to_vector_flt(central_param)
    central_vector = central_vector.to('cpu')
    central_norm = torch.norm(central_vector)

    for clus, vector in clus_represent_parm.items():
        clus_trust[clus] = cos(vector, central_vector)
        clus_normalizing[clus] = central_norm / torch.norm(vector)

    sum_parameters = None
    FLTrustTotalScore = 0
    score_list = []
    for i, local_parameters in enumerate(feature_matrix):
        client_cos = max(clus_trust[cluster_labels[i]].item(), 0)
        client_clipped_value = clus_normalizing[cluster_labels[i]]
        score_list.append(client_cos * client_clipped_value)
        FLTrustTotalScore += client_cos * client_clipped_value
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = client_cos * \
                                      client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * params[i][
                    var]

    if FLTrustTotalScore == 0:
        # print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    # print(score_list)
    return global_parameters




def flame(local_model, update_params, global_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            print(i)
            f.write('\n')
        f.write('\n')
        f.write("--------Round--------")
        f.write('\n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print(benign_client)
   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1

    print(args.wrong_mal, args.right_ben)

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model


def flame_analysis(local_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            f.write('/n')
        f.write('/n')
        f.write("--------Round--------")
        f.write('/n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    return benign_client

def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def lbfgs_torch(args, S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu() #(10,xxxxxx)
    print('------------------------')
    print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu() #(10,xxxxxx)
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
    S_k_time_Y_k = S_k_time_Y_k.cpu()


    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
    S_k_time_S_k = S_k_time_S_k.cpu()
    print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    sigma_k = Y_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1) / (S_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1))
    sigma_k=sigma_k.cpu()
    
    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1,1).cpu()

    approx_prod = sigma_k * v
    print('approx_prod.shape',approx_prod.shape)
    print('v.shape',v.shape)
    print('sigma_k.shape',sigma_k.shape)
    print('sigma_k',sigma_k)
    p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)
    
    approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
    print('approx_prod.shape',approx_prod.shape)

    return approx_prod.T


import sklearn.metrics.pairwise as smp
from scipy.linalg import eigh as largest_eigh
from collections import defaultdict

def roseagg(model_updates, w_glob, args, is_poison=True):
    # === construct a distance matrix ===
    keys = list(model_updates.keys())
    indicative_layer_updates = F.normalize(model_updates[keys[-2]].float())
    K = len(indicative_layer_updates)
    distance_matrix = smp.cosine_similarity(indicative_layer_updates.cpu().numpy()) - np.eye(K)

    # === clustering using dbscan
    partition = DBSCAN(eps=0.2, min_samples=1, metric='cosine').fit(indicative_layer_updates.cpu().numpy())
    clusters = dict()
    print(partition.labels_)
    for i, clu in enumerate(partition.labels_):
        if clu in clusters:
            clusters[clu] += [i]
        else:
            clusters[clu] = [i]

    # === find updates with similar directional contribution ===
    sim_idxs = list()
    for clu in clusters.values():
        if len(clu) > 1:
            sim_idxs.append(clu)

    # === find the master index and remove the other indices within each cluster ===
    remove_idxs = []
    for idxs in sim_idxs:
        idxs = list(idxs)
        idxs.sort()
        if remove_idxs:
            remove_idxs.extend(idxs[1:])
        else:
            remove_idxs = idxs[1:]
    reserve_idxs = list(set(list(range(K))) - set(remove_idxs))
    print(f'Clients with similar directional contribution: {sim_idxs}, '
          f'# Clusters: {max(partition.labels_) + 1}')

    # === partial aggregation. The following process intends to assign a weight to each update ===
    weights = torch.ones(K)
    sims = dict()
    if sim_idxs:
        for sim_idx in sim_idxs:
            sim_idx = list(sim_idx)
            sim_idx.sort()
            sub_distance_matrix = distance_matrix[sim_idx, :][:, sim_idx]
            sub_weights = sub_distance_matrix.sum(axis=0)
            sub_weights = sub_weights / sub_weights.sum()
            for i, idx in enumerate(sim_idx):
                sims[idx] = sub_weights[i]
        for i, s in sims.items():
            weights[i] = s

    # === calculate global update ===
    global_update = defaultdict()
    for name, layer_updates in model_updates.items():
        if 'num_batches_tracked' in name:
            if is_poison:
                num_clients = max(int(args.frac * args.num_users), 1)
                number_of_adversaries = int(args.malicious * num_clients)
                global_update[name] = torch.sum(layer_updates[number_of_adversaries:]) / len(layer_updates[number_of_adversaries])
            else:
                global_update[name] = torch.sum(layer_updates) / len(layer_updates)
        else:
            # === normalization norm ===
            local_norms = np.array([torch.norm(layer_updates[i]).cpu().numpy() for i in range(K)]).reshape(-1, 1)
            if is_poison:
                kmeans = KMeans(n_clusters=2, n_init='auto').fit(local_norms)
                clusters = dict()
                for i, clu in enumerate(kmeans.labels_):
                    if clu in clusters:
                        clusters[clu] += [i]
                    else:
                        clusters[clu] = [i]
                local_norms_0 = local_norms[clusters[0]]
                local_norms_1 = local_norms[clusters[1]]
                if np.median(local_norms_0) > np.median(local_norms_1):
                    normalize_norm = np.median(local_norms_1)
                else:
                    normalize_norm = np.median(local_norms_0)
            else:
                normalize_norm = np.median(local_norms)

            # === plausible clean ingredient ===
            origin_directions = F.normalize(layer_updates)
            origin_directions = weights.view(-1, 1).to(args.device) * origin_directions

            first_idxs = []
            if sim_idxs:
                # === targeted aggregation within each cluster ===
                for idxs in sim_idxs:
                    idxs = list(idxs)
                    idxs.sort()
                    first_idx = idxs[0]
                    first_idxs.append(first_idx)
                    for idx in idxs[1:]:
                        origin_directions[first_idx] = origin_directions[first_idx] + origin_directions[idx]
                    origin_directions[first_idx] /= torch.norm(origin_directions[first_idx])
                origin_directions = origin_directions[reserve_idxs]

            # === extract plausible clean ingredient ===
            N = origin_directions.size(0)
            X = torch.matmul(origin_directions, origin_directions.T)
            evals_large, evecs_large = largest_eigh(X.detach().cpu().numpy(), eigvals=(N - N, N - 1))
            evals_large = torch.tensor(evals_large)[-1].to(args.device)
            evecs_large = torch.tensor(evecs_large)[:, -1].to(args.device)
            principal_direction = torch.matmul(evecs_large.view(1, -1), origin_directions).T / torch.sqrt(
                evals_large)

            # === reweight partial aggregated model udpates ===
            new_weights = torch.pow(torch.matmul(principal_direction.view(1, -1), origin_directions.T), 2)
            new_weights = new_weights / new_weights.sum()

            # === aggregation ===
            origin_directions += torch.normal(0, 0.003, origin_directions.size()).to(args.device)
            origin_directions = F.normalize(origin_directions)
            scale = normalize_norm
            principal_direction = torch.matmul(new_weights, origin_directions * scale)
            global_update[name] = principal_direction

    for name, param in w_glob.items():
        w_glob[name] = param.data + global_update[name].view(param.size())

    return w_glob

def convert_w_updates_to_model_updates(w_updates):
    # 首先获取所有可能的参数名称
    all_param_names = set()
    for client_update in w_updates:
        all_param_names.update(client_update.keys())

    model_updates = {}
    for param_name in all_param_names:
        # 收集所有客户端该参数的更新
        param_updates = []
        for client_update in w_updates:
            if param_name in client_update:
                param_updates.append(client_update[param_name].view(1, -1))
            else:
                # 如果某个客户端没有该参数的更新，用全零张量代替
                shape = next(iter(client_update.values())).view(1, -1).shape
                param_updates.append(torch.zeros(shape))

        # 将所有客户端的更新堆叠成一个张量
        model_updates[param_name] = torch.cat(param_updates, dim=0)

    return model_updates