import sys
sys.path.append("../")
import time

import torch
import copy

class A3FL:
    def __init__(self, target_class, bkd_ratio, triggerY, triggerX, trigger_size, device):
        self.device = device
        self.previous_global_model = None
        self.setup(triggerY, triggerX, trigger_size)
        self.target_class = target_class
        self.bkd_ratio = bkd_ratio
        self.start = 0

    def setup(self, triggerY, triggerX, trigger_size):
        self.handcraft_rnds = 0
        self.trigger = torch.randn((1,3,32,32), requires_grad=False, device=self.device)*0.5
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, triggerY:triggerY + trigger_size, triggerX:triggerX + trigger_size] = 1
        self.mask = self.mask.to(self.device)
        self.trigger0 = self.trigger.clone()

    def init_badnets_trigger(self):
        print('Setup baseline trigger pattern.')
        self.trigger[:, 0, :,:] = 1
        return
    
    def get_adv_model(self, model, dl, trigger, mask): # 用后门数据对抗训练模型，返回对抗训练模型和对抗训练后的模型和原始模型的余弦相似度
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(5):
            for inputs, labels in dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = trigger * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            #if 'conv' in name:
            sim_count += 1
            sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    def search_trigger(self, model, dl):
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        adv_models = []
        adv_ws = []

        
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = 0.01
        
        K = 10
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
        for iter in range(K):

            if iter % 1 == 0 and iter != 0:
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(1):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)


            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = t*m +(1-m)*inputs
                labels[:] = self.target_class
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = 0.01*adv_w*nm_loss
                        else:
                            loss += 0.01*adv_w*nm_loss
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -2, max = 2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()



    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.bkd_ratio * inputs.shape[0])
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        labels[:bkd_num] = self.target_class
        return inputs, labels
    