"""Trainer

    Train all your model here.
"""

import torch
import datetime
import time
import os
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score


from utils.function import init_logging, init_environment, get_lr, \
    print_loss_sometime
from utils.metric import mean_class_recall
import config
import dataset
import model
from loss import class_balanced_loss

class trainer():
    def __init__(self):
        self.configs = config.Config()
        configs_dict = self.configs.get_config()
        # Load hyper parameter from config file
        self.disable_save = True
        self.exp = configs_dict["experiment_index"]
        self.cuda_ids = configs_dict["cudas"]
        self.num_workers = configs_dict["num_workers"]
        self.seed = configs_dict["seed"]
        self.n_epochs = configs_dict["n_epochs"]
        log_dir = configs_dict["log_dir"]
        self.model_dir = configs_dict["model_dir"]
        self.batch_size = configs_dict["batch_size"]
        self.learning_rate = configs_dict["learning_rate"]
        self.backbone = configs_dict["backbone"]
        self.eval_frequency = configs_dict["eval_frequency"]
        self.resume = configs_dict["resume"]
        self.optimizer = configs_dict["optimizer"]
        self.initialization = configs_dict["initialization"]
        self.num_classes = configs_dict["num_classes"]
        self.iter_fold = configs_dict["iter_fold"]
        self.loss_fn = configs_dict["loss_fn"]
        self.backbone = configs_dict["backbone"]
        self._print = init_logging(log_dir, self.exp).info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tf_log = os.path.join(log_dir, self.exp)
        self.writer = SummaryWriter(log_dir=tf_log)
        self.net = None
        self.epoch = 99

    def train(self):
        init_environment(seed=self.seed, cuda_id=self.cuda_ids)
        self.configs.print_config(self._print)
        # Pre-peocessed input image
        if self.backbone in ["resnet50", "resnet18"]:
            re_size = 300
            input_size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif self.backbone in ["NASNetALarge", "PNASNet5Large"]:
            re_size = 441
            input_size = 331
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            self._print("Need backbone")
            sys.exit(-1)

        self._print("=> Image resize to {} and crop to {}".format(re_size, input_size))

        self.train_transform = transforms.Compose([
            transforms.Resize(re_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                    scale=[0.7, 1.3]),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        input_channel = 3
        print(f'using iter_fold {self.iter_fold}')
        trainset = dataset.Skin7(root="./data/", iter_fold=self.iter_fold, train=True,
                                 transform=self.train_transform)
        valset = dataset.Skin7(root="./data/", iter_fold=self.iter_fold, train=False,
                               transform=self.val_transform)

        self.net = model.Network(backbone=self.backbone, num_classes=self.num_classes,
                            input_channel=input_channel, pretrained=self.initialization)

        self._print("=> Using device ids: {}".format(self.cuda_ids))
        device_ids = list(range(len(self.cuda_ids.split(","))))
        train_sampler = val_sampler = None
        if len(device_ids) == 1:
            self._print("Model single cuda")
            self.net = self.net.to(self.device)
        else:
            self._print("Model parallel !!")
            # torch.distributed.init_process_group(backend="nccl")
            # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            # val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
            # net = torch.nn.parallel.DistributedDataParallel(net)
            self.net = nn.DataParallel(self.net, device_ids=device_ids).to(self.device)

        self._print("=> self.iter_fold is {}".format(self.iter_fold))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, pin_memory=True,
                                                  num_workers=self.num_workers,
                                                  sampler=train_sampler)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=self.num_workers,
                                                sampler=val_sampler)

        # Loss
        if self.loss_fn == "WCE":
            self._print("Loss function is WCE")
            weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
            class_weights = torch.FloatTensor(weights)
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        elif self.loss_fn == "CE":
            self._print("Loss function is CE")
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self._print("Need loss function.")

        # Optmizer
        scheduler = None
        if self.optimizer == "SGD":
            self._print("=> Using self.optimizer SGD with lr:{:.4f}".format(self.learning_rate))
            opt = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt, mode='min', factor=0.1, patience=50, verbose=True,
                        threshold=1e-4)
        elif self.optimizer == "Adam":
            self._print("=> Using self.optimizer Adam with lr:{:.4f}".format(self.learning_rate))
            opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate,
                                   betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
        else:
            self._print("Need self.optimizer")
            sys.exit(-1)

        start_epoch = 0
        if self.resume is not None and self.resume != "":
            self._print("=> Resume from model at epoch {}".format(self.resume))
            self.resume_path = os.path.join(self.model_dir, str(self.exp), str(f'Task3_{self.backbone}_{self.exp}_{self.resume}.pt'))
            ckpt = torch.load(self.resume_path)
            self.net.load_state_dict(ckpt)
            start_epoch = self.resume + 1
        else:
            self._print("Train from scrach!!")

        desc = "Exp-{}-Train".format(self.exp)
        sota = {}
        sota["epoch"] = start_epoch
        sota["mcr"] = -1.0

        for epoch in range(start_epoch+1, self.n_epochs+1):
            self.epoch = epoch
            self.net.train()
            losses = []
            for batch_idx, (data, target) in enumerate(trainloader):
                start = datetime.datetime.now()
                print(f'epoch {epoch}, batch {batch_idx}, start = {start}')
                data, target = data.to(self.device), target.to(self.device)
                predict = self.net(data)
                print('predict done')
                opt.zero_grad()
                loss = criterion(predict, target)
                loss.backward()
                print('backward done')
                opt.step()
                print(f'time = {datetime.datetime.now() - start}\n')
                losses.append(loss.item())

            print(f'self.eval_frequency = {self.eval_frequency}')
            if epoch != 0 and epoch % self.eval_frequency == 0:
                print(f'evaling')
                # print to log
                dicts = {
                    "epoch": epoch, "n_epochs": self.n_epochs, "loss": loss.item()
                }
                # print_loss_sometime(dicts, _print=self._print)
                print(f'loss = {loss.item()}')

                train_avg_loss = np.mean(losses)
                if scheduler is not None:
                    scheduler.step(train_avg_loss)

                self.writer.add_scalar("Lr", get_lr(opt), epoch)
                self.writer.add_scalar("Loss/train/", train_avg_loss, epoch)

                self.net.eval()
                y_true = []
                y_pred = []
                for _, (data, target) in enumerate(trainloader):
                    data = data.to(self.device)
                    predict = torch.argmax(self.net(data), dim=1).cpu().data.numpy()
                    y_pred.extend(predict)
                    target = target.cpu().data.numpy()
                    y_true.extend(target)

                acc = accuracy_score(y_true, y_pred)
                mcr = mean_class_recall(y_true, y_pred)
                self._print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
                self._print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
                self.writer.add_scalar("Acc/train/", acc, epoch)
                self.writer.add_scalar("Mcr/train/", mcr, epoch)

                y_true = []
                y_pred = []
                for _, (data, target) in enumerate(valloader):
                    data = data.to(self.device)
                    predict = torch.argmax(self.net(data), dim=1).cpu().data.numpy()
                    y_pred.extend(predict)
                    target = target.cpu().data.numpy()
                    y_true.extend(target)

                acc = accuracy_score(y_true, y_pred)
                mcr = mean_class_recall(y_true, y_pred)
                self._print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
                self._print("=> Epoch:{} - val mcr: {:.4f}".format(epoch, mcr))
                self.writer.add_scalar("Acc/val/", acc, epoch)
                self.writer.add_scalar("Mcr/val/", mcr, epoch)

                # Val acc
                if mcr > sota["mcr"]:
                    sota["mcr"] = mcr
                    sota["epoch"] = epoch
                    model_path = os.path.join(self.model_dir, str(self.exp), str(f'Task3_{self.backbone}_{self.exp}_{epoch}.pt'))
                    if os.path.exists(model_path):
                        model_path = os.path.join(self.model_dir,
                                                  str(self.exp),
                                                  str(f'Task3_{self.backbone}_{self.exp}_{epoch}_{time.time()}.pt'))
                    self._print("=> Save model in {}".format(model_path))
                    net_state_dict = self.net.state_dict()
                    if not self.disable_save:
                        self._print(f'SAVING TO {model_path}')
                        torch.save(net_state_dict, model_path)
                    else:
                        self._print(f'SAVING IS DISABLED! MAKE SURE TO SAVE!')

        self._print("=> Finish Training")
        self._print("=> Best epoch {} with {} on Val: {:.4f}".format(sota["epoch"],
                                                                "sota",
                                                                sota["mcr"]))
    def write(self, model_path):
        self.net.eval()
        if not model_path:
            model_path = os.path.join(self.model_dir, str(self.exp), str(f'Task3_{self.backbone}_{self.exp}_{self.epoch}.pt'))
        if os.path.exists(model_path):
            model_path = os.path.join(self.model_dir,
                                      str(self.exp),
                                      str(f'Task3_{self.backbone}_{self.exp}_{self.epoch}_{time.time()}.pt'))
        net_state_dict = self.net.state_dict()
        torch.save(net_state_dict, model_path)

    def load(self, path):
        self.net = model.Network(backbone=self.backbone, num_classes=self.num_classes,
                                 input_channel=3, pretrained=self.initialization)
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt)

    def test(self, count=10):
        self.net.eval()
        import pandas as pd
        csvfile = pd.read_csv('../ISIC2018_Task3_Test_Input/ISIC2018_Task3_Test_GroundTruth.csv')
        raw_data = csvfile.values

        with torch.no_grad():
            for idx in np.random.choice(raw_data.shape[0], count, replace=True):
                data = raw_data[idx][0]
                try:
                    target = np.where(raw_data[idx][1:] == 1.0)[0][0]
                except Exception as e:
                    print(f'err getting target for {data}, raw = {raw_data}')
                    print(repr(e))
                    continue
                img = dataset.pil_loader(f'../ISIC2018_Task3_Test_Input/{data}.jpg')
                img = self.val_transform(img)
                img = torch.unsqueeze(img, 0)
                predict = torch.argmax(self.net(img), dim=1).cpu().data.numpy()
                if predict == target:
                    print(f'{data} correctly predicted {predict}')
                else:
                    print(f'{data} INCORRECTLY predicted {predict}, true answer = {target}')

if __name__ == "__main__":
    trainer().train()
