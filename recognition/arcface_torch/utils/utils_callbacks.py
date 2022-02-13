import logging
import os
import time
from typing import List

import torch

from eval import verification, spoofing_verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
import cv2
import numpy as np


class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 liveness_loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                    self.writer.add_scalar('liveness_loss', liveness_loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   Liveness loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, liveness_loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   Liveness loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, liveness_loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                liveness_loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

class CallBackSpoofing(object):
    def __init__(self, rec_prefix, image_size=(112, 112)):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        if self.rank == 0:
            self.init_dataset(data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        acc = spoofing_verification.test(
            self.dataset, backbone, 10)
        logging.info('[CelebA_Spoof][%d] Spoofing accuracy: %1.5f' % (global_step, acc))
        if acc > self.highest_acc:
            self.highest_acc = acc
        logging.info(
            '[CelebA_Spoof][%d]Accuracy-Highest: %1.5f' % (global_step, self.highest_acc))

    def init_dataset(self, data_dir, image_size):
        images = []
        liveness_labels = []
        test_dir = os.path.join(data_dir, 'Norm_data/test')
        for subdir in os.listdir(test_dir):
            live_dir = os.path.join(test_dir, subdir, 'live')
            if os.path.isdir(live_dir):
                liveness_label = 1
                for file_name in os.listdir(live_dir):
                    if '.jpg' in file_name or '.png' in file_name:
                        file_path = os.path.join(live_dir,file_name)
                        im = cv2.imread(file_path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = np.transpose(im, axes=(2, 0, 1))
                        im_tensor = torch.from_numpy(im)
                        images.append(im_tensor)
                        liveness_labels.append(liveness_label)
            spoof_dir = os.path.join(test_dir, subdir, 'spoof')
            if os.path.isdir(spoof_dir):
                liveness_label = 0
                for file_name in os.listdir(spoof_dir):
                    if '.jpg' in file_name or '.png' in file_name:
                        file_path = os.path.join(spoof_dir,file_name)
                        im = cv2.imread(file_path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = np.transpose(im, axes=(2, 0, 1))
                        im_tensor = torch.from_numpy(im).float()
                        images.append(im_tensor)
                        liveness_labels.append(liveness_label)
        images = torch.stack(images, dim=0)
        self.dataset = [images, liveness_labels]

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()