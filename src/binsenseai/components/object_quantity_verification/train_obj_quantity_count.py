import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import data_utils_obj_quantity
import numpy as np
from PIL import Image
import json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ABID verification')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=10, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')
parser.add_argument('--max-target', default=20, type=int, metavar='N', help='maximum number of target images for validation (default: 20)')




best_prec = 0
train_loss_list = []
val_acc_list = []
train_acc_list = []
def main():
    global args, best_prec, train_loss_list, val_acc_list
    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.arch))
    # net = models.__dict__[args.arch]
    
    # in_features = net.fc.in_features
    # net.fc = nn.Linear(in_features,1)
    # net.to(device)
    
    net = models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 50)
    net.to(device)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            train_loss_list = checkpoint['train_loss_list']
            val_acc_list = checkpoint['val_acc_list']
            params = net.parameters()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    params = net.parameters()
    snapshot_fname = "snapshots/%s.pth.tar" % args.arch
    snapshot_best_fname = "snapshots/%s_best.pth.tar" % args.arch

     
    # Disable cudnn.benchmark (for CPU usage) cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    quant_train_json_file = "/home/ubuntu/abid/dataset/obj_num_verification_train.json"
    quant_val_json_file = "/home/ubuntu/abid/dataset/obj_num_verification_val.json"
    
    train_loader = torch.utils.data.DataLoader(
        data_utils_obj_quantity.ImageFolderTraining(args.data,quant_train_json_file , transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        data_utils_obj_quantity.ImageFolderValidation (args.data,quant_val_json_file , args.max_target, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    

    # Create the criterion instance
    criterion = nn.CrossEntropyLoss()

    # Move the criterion to the desired device
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """ optimizer = torch.optim.Adam(params, args.lr,
                                weight_decay=args.weight_decay) """
    

    # evaluate on validation set
    if args.evaluate:
        validate(val_loader, net, criterion, True)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch)

    # evaluate on validation set
        
        prec = validate(val_loader, net, criterion, False)

        # remember best prec@1 and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        torch.save({ 
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_prec': best_prec,
            'train_loss_list': train_loss_list,
            'val_acc_list': val_acc_list,
            'model': net ##added newly
        }, snapshot_fname)
        
        if is_best:
            shutil.copyfile(snapshot_fname, snapshot_best_fname) 
        


# Updated train function 
def train(train_loader, model, criterion, optimizer, epoch):
    cur_lr = adjust_learning_rate(optimizer, epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Move tensors to the specified device
        #target=target.view(-1, 1)
        target = target.to(device)
        input = input.to(device)
        #print('213 Move tensors to the specified device:target,input',target.shape,input.shape)
        #print('214 Move tensors to the specified device:target,input',target)
        # Clear accumulated gradients
        #optimizer.zero_grad()

        # Forward pass
        output = model(input)
        #print('220 output,target.view(-1, 1)',output,target)
        # Compute loss
        loss = criterion(output, target.long())  # Reshape target to match output size

        max_elements,max_idxs = torch.max(output.detach(),1)
        correct = (target==max_idxs)
        acc = float(correct.sum())/input.size(0)
        #print('227 acc, correct.sum(),input.size(0) ',acc,correct.sum(),input.size(0))
        #print('Epoch: [{0}][{1}'.format(max_elements,max_idxs))
        #losses.update(loss.item(), input.size(0))
        #train_acc.update(acc, input.size(0))

        losses.update(loss.item(), input.size(0))
        train_acc.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Backward pass
        #loss.backward()
        
        # Update weights
        #optimizer.step()
        #print('242 target, output',target, output)
        # Compute accuracy
        #accuracy = (target==output) #(output > 0.5).eq(target).float().mean()
        #print('245 Compute accuracy output target  loss ', output, target, loss)


        # Record statistics
        #losses.update(loss.item(), input.size(0))
        #train_acc.update(accuracy.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}] lr {cur_lr:.5f}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec {train_acc.val:.3f} ({train_acc.avg:.3f})'.format(
               epoch, i, len(train_loader), cur_lr=cur_lr, batch_time=batch_time,
               data_time=data_time, loss=losses, train_acc=train_acc))
        train_loss_list.append(losses.val)
        
        
        train_acc_list.append(train_acc.avg)
        #print(train_acc_list)
def validate(val_loader, model, criterion, file_out):
    batch_time = AverageMeter()
    val_acc = AverageMeter()
    losses = AverageMeter()
    
    correct = 0 
    total = 0

    # switch to evaluate mode
    model.eval()

    if file_out:
        f = open('obj_quantity_count_result.txt','w') 

    end = time.time()
    
    
    for i, (input, target) in enumerate(val_loader):
        # input = input.view(-1, 3, 224, 224)
        # target = target.view(-1)

        # Move tensors to the specified device
        target = target.to(device)
        input = input.to(device)


        with torch.no_grad():
            input_var = input
            target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var.long())
        _, predicted =  torch.max(output.detach(),1)
        correct = (target==predicted)
        #print('correct.sum(),input.size(0)',correct.sum(),input.size(0))
        acc = float(correct.sum())/input.size(0)
        
        # Compute loss

      
        # measure accuracy and record loss
        # n_target = input.size(0)
        # n_pos = output.gt(0.5).sum()
        # pred = (n_pos / (n_target * 1.0)) > 0.5
        # correct = pred == target[0]
        # val_acc.update(correct.item(), 1)
        
         # measure accuracy and record loss

        if file_out:
            for j in range(input.size(0)):
                f.write('%d\n' % predicted[j])
            

        # measure accuracy and record loss
        #losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input.size(0))
        val_acc.update(acc, input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print('Test: [{0}/{1}] n_pos/n_tar ({n_pos:d}/{n_target:d})\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Prec {val_acc.val:.3f} ({val_acc.avg:.3f})'.format(
        #         i, len(val_loader), n_pos=n_pos, n_target=n_target,
        #         batch_time=batch_time,
        #         val_acc=val_acc))
        
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec {val_acc.val:.3f} ({val_acc.avg:.3f})'.format(
               i, len(val_loader), batch_time=batch_time, loss=losses,
               val_acc=val_acc))

    
    print(' * Prec {val_acc.avg:.3f}'.format(val_acc=val_acc))
    if file_out:
        f.close()

    val_acc_list.append(val_acc.avg)
    return val_acc.avg
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lrd epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrd))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()