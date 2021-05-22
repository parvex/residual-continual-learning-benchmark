import copy
from enum import Enum
from torch.autograd import Variable
import torch
import random
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from models.combined import CombinedResNet
from .default import NormalNN
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from models import combined


class L2(NormalNN):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(self, agent_config):
        super(L2, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
        # False: Each task has its own importance matrix and model parameters

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

    def learn_batch(self, train_loader, val_loader=None):

        self.log('#reg_term:', len(self.regularization_terms))

        # 1.Learn the parameters for current task
        super(L2, self).learn_batch(train_loader, val_loader)

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance': importance, 'task_param': task_param}

    def criterion(self, inputs, targets, tasks, regularization=True, **kwargs):
        loss = super(L2, self).criterion(inputs, targets, tasks, **kwargs)

        if regularization and len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss += self.config['reg_coef'] * reg_loss
        return loss


class EWC(L2):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(self, agent_config):
        super(EWC, self).__init__(agent_config)
        self.online_reg = False
        self.n_fisher_sample = None
        self.empFI = False

    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        self.log('Computing EWC')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms) > 0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        # Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            self.log('Sample', self.n_fisher_sample, 'for estimating the F matrix.')
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=1)

        mode = self.training
        self.eval()

        # Accumulate the square of gradients
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input)

            # Sample the labels for estimating the gradients
            # For multi-headed model, the batch of data will be from the same task,
            # so we just use task[0] as the task name to fetch corresponding predictions
            # For single-headed model, just use the max of predictions from preds['All']
            task_name = task[0] if self.multihead else 'All'

            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,
                                                                                    :self.valid_out_dim]
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # - Alternative ind by multinomial sampling. Its performance is similar. -
            # prob = torch.nn.functional.softmax(preds['All'],dim=1)
            # ind = torch.multinomial(prob,1).flatten()

            if self.empFI:  # Use groundtruth label (default is without this)
                ind = target

            loss = self.criterion(preds, ind, task, regularization=False)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += ((self.params[n].grad ** 2) * len(input) / len(dataloader))

        self.train(mode=mode)

        return importance


def EWC_online(agent_config):
    agent = EWC(agent_config)
    agent.online_reg = True
    return agent


class SI(L2):
    """
    @inproceedings{zenke2017continual,
        title={Continual Learning Through Synaptic Intelligence},
        author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
        booktitle={International Conference on Machine Learning},
        year={2017},
        url={https://arxiv.org/abs/1703.04200}
    }
    """

    def __init__(self, agent_config):
        super(SI, self).__init__(agent_config)
        self.online_reg = True  # Original SI works in an online updating fashion
        self.damping_factor = 0.1
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

    def update_model(self, inputs, targets, tasks):

        unreg_gradients = {}

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks, regularization=False)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss.detach(), out

    """
    # - Alternative simplified implementation with similar performance -
    def update_model(self, inputs, targets, tasks):
        # A wrapper of original update step to include the estimation of w

        # Backup prev param if not done yet
        # The backup only happened at the beginning of a new task
        if len(self.prev_params) == 0:
            for n, p in self.params.items():
                self.prev_params[n] = p.clone().detach()

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2.Calculate the loss as usual
        loss, out = super(SI, self).update_model(inputs, targets, tasks)

        # 3.Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if p.grad is not None:  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= p.grad * delta  # w[n] is >=0

        return loss.detach(), out
    """

    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg, 'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms) > 0:  # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
            self.w[n].zero_()

        return importance


class MAS(L2):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, agent_config):
        super(MAS, self).__init__(agent_config)
        self.online_reg = True

    def calculate_importance(self, dataloader):
        self.log('Computing MAS')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms) > 0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        mode = self.training
        self.eval()

        # Accumulate the gradients of L2 loss on the outputs
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input)

            # Sample the labels for estimating the gradients
            # For multi-headed model, the batch of data will be from the same task,
            # so we just use task[0] as the task name to fetch corresponding predictions
            # For single-headed model, just use the max of predictions from preds['All']
            task_name = task[0] if self.multihead else 'All'

            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the  loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,
                                                                                    :self.valid_out_dim]

            pred.pow_(2)
            loss = pred.mean()

            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += (self.params[n].grad.abs() / len(dataloader))

        self.train(mode=mode)

        return importance


#################################ZZSN

class ResCL(NormalNN):
    """
    @misc{lee2020residual,
          title={Residual Continual Learning},
          author={Janghyeon Lee and Donggyu Joo and Hyeong Gwon Hong and Junmo Kim},
          year={2020},
          eprint={2002.06774},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
    """

    def __init__(self, agent_config: dict):
        super(ResCL, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_count = 0
        # self.learning_state = LearningState.INIT
        self.lambd_dec = 0.0001
        self.lambd = agent_config['lambd']
        self.source_model = None
        self.target_model = None
        self.source_pred = None
        self.target_pred = None
        self.split_size = agent_config['split_size']

    def learn_batch(self, train_loader: DataLoader, val_loader: DataLoader = None) -> None:

        if self.task_count == 0:
            self.log("ResCL learning init model")
            # 1.Learn the parameters for 1. as normal ResNet
            del self.criterion_fn
            self.criterion_fn = nn.CrossEntropyLoss()
            super(ResCL, self).learn_batch(train_loader, val_loader)
        else:
            # Residual Continual Learning algorithm
            self.residual_continual_learning(train_loader, val_loader)

        self.task_count += 1

    def residual_continual_learning(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.source_model = self.model
        self.target_model = copy.deepcopy(self.model)

        # Now target_model will be fine tuned so learning model is changed
        self.log("ResCL fine tuning target model")
        del self.criterion_fn
        self.criterion_fn = self.fine_tuning_loss
        self.model = self.target_model
        self.move_to_device()
        super(ResCL, self).learn_batch(train_loader, val_loader)

        self.log("ResCL learning combined source and target model")
        del self.criterion_fn
        self.criterion_fn = self.combined_learn_loss
        self.model = CombinedResNet(self.source_model, self.target_model, self.config['out_dim']['All'], self.gpu, self.valid_out_dim)
        self.move_to_device()
        super(ResCL, self).learn_batch(train_loader, val_loader)

        self.model = self.model.get_combined_network()

    # override update_model to calculate out for source and target models to calculate loss
    def update_model(self, inputs: Tensor, targets: Tensor, tasks: tuple) -> tuple:
        out = self.forward(inputs)
        if self.criterion_fn == self.combined_learn_loss:
            self.source_pred = self.source_model.forward(inputs)['All'][:, :self.valid_out_dim]
            self.target_pred = self.target_model.forward(inputs)['All'][:, :self.valid_out_dim]

        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def fine_tuning_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        # take few last columns (new task specific)
        logsoftmax_preds = F.log_softmax(pred[:, -self.split_size:] / 2, dim=1)
        target_probs = torch.nn.functional.one_hot(target).to(torch.float32)[:, -self.split_size:]
        l_kl = F.kl_div(logsoftmax_preds, target_probs, reduction='sum') * (2 ** 2) / target.shape[0]
        l2_reg = self.calculate_l2()
        return l_kl + l2_reg

    def combined_learn_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        combined_source_preds = F.softmax(pred[:, :(self.valid_out_dim - self.split_size)] / 2, dim=1)
        combined_target_preds = F.softmax(pred[:, -self.split_size:] / 2, dim=1)
        source_ls_preds = F.log_softmax(self.source_pred[:, :(self.valid_out_dim - self.split_size)] / 2, dim=1)
        target_ls_preds = F.log_softmax(self.target_pred[:, -self.split_size:] / 2, dim=1)

        l_kl_s = F.kl_div(source_ls_preds, combined_source_preds, reduction='sum') * (2 ** 2) / target.shape[0]
        l_kl_t = F.kl_div(target_ls_preds, combined_target_preds, reduction='sum') * (2 ** 2) / target.shape[0]
        l2_reg = self.calculate_l2()
        alfa_l1_reg = self.calculate_alfa_l1()
        return l_kl_s + l_kl_t + l2_reg + alfa_l1_reg

    def calculate_l2(self) -> Tensor:
        l2_reg = torch.tensor(0.)
        if self.gpu:
            l2_reg = l2_reg.cuda()
        for param in self.model.parameters():
            l2_reg += torch.norm(param, p=2)
        return l2_reg * self.lambd_dec

    def calculate_alfa_l1(self) -> Tensor:
        l1_reg = torch.tensor(0.)
        if self.gpu:
            l1_reg = l1_reg.cuda()
        for param in self.model.alfa_source.items():
            l1_reg += torch.norm(param[1], p=1)
        for param in self.model.alfa_target.items():
            l1_reg += torch.norm(param[1], p=1)
        return l1_reg * self.lambd

    def move_to_device(self) -> None:
        if self.gpu:
            self.model.cuda()
