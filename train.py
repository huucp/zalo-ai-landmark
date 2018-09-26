from __future__ import division

from zalo_utils import *
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import argparse
import copy
from torchvision import transforms
import os, sys
from time import time, strftime
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Zalo Landmark Identification Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', choices=['resnet', 'inception_v3', 'xception'], type=str,
                    help='model')
parser.add_argument('--depth', default=101, choices=[18, 34, 50, 101, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type=str, help='optimizer')
parser.add_argument('--model_path', type=str, default=' ')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--num_epochs', default=86, type=int,
                    help='Number of epochs in training')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--check_after', default=2,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1, 2],
                    # 0: from scratch, 1: from pretrained Resnet, 2: specific checkpoint in model_path
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")
parser.add_argument('--frozen_until', '-fu', type=int, default=6,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float,
                    help="number of training samples per class")
parser.add_argument('--random_state', '-rs', type=int, default=42)
parser.add_argument('--flip', type=float, default=.2)
parser.add_argument('--brightness', type=float, default=.05)
parser.add_argument('--contrast', type=float, default=.05)
parser.add_argument('--saturation', type=float, default=.05)
parser.add_argument('--hue', type=float, default=.05)

parser.add_argument('--skip', type=str, default='', help='List of label to skip')

args = parser.parse_args()

KTOP = 3  # top k error
global_step = 0
writer = SummaryWriter()


def exp_lr_scheduler(args, optimizer, epoch):
    # after epoch 100, no more learning rate decay
    init_lr = args.lr
    lr_decay_epoch = 8  # decay lr after each 10 epoch
    lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch))
    weight_decay = args.weight_decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr


use_gpu = torch.cuda.is_available()

json_file = '../data/train_val2018.json'
data_dir = '../data/TrainVal/'

print('Loading data')
fns, lbs, cnt = get_fns_lbs(data_dir, json_file)

print('Total files in the original dataset: {}'.format(cnt))
print('Total files with > 0 byes: {}'.format(len(fns)))
print('Total files with zero bytes {}'.format(cnt - len(fns)))

if args.skip != '':
    filtered_fns = []
    filtered_lbs = []
    skip_lbs = set(map(int, args.skip.split(',')))
    print('Skip lbs ' + args.skip)
    for i in range(len(fns)):
        if lbs[i] not in skip_lbs:
            filtered_fns.append(fns[i])
            filtered_lbs.append(lbs[i])
    fns = filtered_fns
    lbs = filtered_lbs

print('Total image after removing skip labels: {}'.format(len(fns)))


############################
print('Split data')
train_fns, val_fns, train_lbs, val_lbs = train_test_split(fns, lbs, test_size=args.val_ratio, random_state=12)
print('Number of training imgs: {}'.format(len(train_fns)))
print('Number of validation imgs: {}'.format(len(val_fns)))

########### 
print('DataLoader ....')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Train -> Preprocessing -> Tensor

# Pin memory
if torch.cuda.is_available():
    pin_memory = True
else:
    pin_memory = False

if args.net_type.startswith('inception') or args.net_type.startswith('xception'):
    scale_size = 333
    input_size = 299
else:
    scale_size = 256
    input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(p=args.flip),  # simple data augmentation
        transforms.ColorJitter(brightness=args.brightness,
                               contrast=args.contrast,
                               saturation=args.saturation,
                               hue=args.hue),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

dsets = dict()
dsets['train'] = LandmarkDataSet(train_fns, train_lbs, transform=data_transforms['train'])
dsets['val'] = LandmarkDataSet(val_fns, val_lbs, transform=data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=args.batch_size,
                                   shuffle=(x != 'val'),
                                   num_workers=args.num_workers,
                                   pin_memory=(x != 'val'))
    for x in ['train', 'val']
}
########## 
print('Load model')

prefix = args.net_type
if args.net_type == 'resnet':
    prefix = args.net_type + '-{}'.format(args.depth)

saved_model_fn = prefix + '_' + strftime('%m%d_%H%M')
old_model = './checkpoint/' + prefix + '_' + args.model_path + '.t7'
num_classes = 103
if args.train_from == 2 and os.path.isfile(old_model):
    print("| Load pretrained at  %s..." % old_model)
    checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
    tmp = checkpoint['model']
    model = unparallelize_model(tmp)
    best_top3 = checkpoint['top3']
    print('previous top3\t%.4f' % best_top3)
    print('=============================================')
elif args.net_type == 'resnet':
    model = MyResNet(args.depth, num_classes)
elif args.net_type == 'inception_v3':
    model = MyInception(num_classes)
elif args.net_type == 'xception':
    model = MyXception(num_classes)

##################
print('Start training ... ')
criterion = nn.CrossEntropyLoss()
model, optimizer = net_frozen(args, model)
model = parallelize_model(model)

N_train = len(train_lbs)
N_valid = len(val_lbs)
best_top3 = 1
t0 = time()
for epoch in range(args.num_epochs):
    optimizer, lr = exp_lr_scheduler(args, optimizer, epoch)
    print('#################################################################')
    print('=> Training Epoch #%d, LR=%.10f' % (epoch + 1, lr))
    # torch.set_grad_enabled(True)

    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    running_loss_src, running_corrects_src, tot_src = 0.0, 0.0, 0.0
    runnning_topk_corrects = 0.0
    ########################
    model.train()
    torch.set_grad_enabled(True)
    ## Training
    # local_src_data = None
    for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['train']):
        optimizer.zero_grad()
        # bs, ncrops, c, h, w = inputs.size()
        inputs = cvt_to_gpu(inputs)
        labels = cvt_to_gpu(labels)
        # outputs = model(inputs.view(-1, c, h, w))
        outputs = model(inputs)

        if isinstance(outputs, tuple):
            outputs, _ = outputs

        # outputs = outputs.view(bs, ncrops, -1).mean(1)

        loss = criterion(outputs, labels)
        running_loss += loss * inputs.shape[0]
        loss.backward()
        optimizer.step()
        ############################################
        _, preds = torch.max(outputs.data, 1)
        # topk
        top3correct, _ = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
        runnning_topk_corrects += top3correct
        # pdb.set_trace()
        running_loss += loss.item()
        running_corrects += preds.eq(labels.data).cpu().sum()
        tot += labels.size(0)
        sys.stdout.write('\r')
        try:
            batch_loss = loss.item()
        except NameError:
            batch_loss = 0

        top1error = 1 - float(running_corrects) / tot
        top3error = 1 - float(runnning_topk_corrects) / tot
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f'
                         % (epoch + 1, args.num_epochs, batch_idx + 1,
                            (len(train_fns) // args.batch_size), batch_loss / args.batch_size,
                            top1error, top3error))
        sys.stdout.flush()
        sys.stdout.write('\r')
        writer.add_scalar('batch_loss', batch_loss, ++global_step)

    top1error = 1 - float(running_corrects) / N_train
    top3error = 1 - float(runnning_topk_corrects) / N_train
    epoch_loss = running_loss / N_train
    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f' \
          % (epoch_loss, top1error, top3error))

    print_eta(t0, epoch, args.num_epochs)

    ###################################
    ## Validation
    if (epoch + 1) % args.check_after == 0:
        # Validation
        running_loss, running_corrects, tot = 0.0, 0.0, 0.0
        runnning_topk_corrects = 0
        torch.set_grad_enabled(False)
        model.eval()
        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
            inputs = cvt_to_gpu(inputs)
            labels = cvt_to_gpu(labels)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs, _ = outputs

            _, preds = torch.max(outputs.data, 1)
            top3correct, top3error = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
            runnning_topk_corrects += top3correct
            running_loss += loss.item()
            running_corrects += preds.eq(labels.data).cpu().sum()
            tot += labels.size(0)

        epoch_loss = running_loss / N_valid
        top1error = 1 - float(running_corrects) / N_valid
        top3error = 1 - float(runnning_topk_corrects) / N_valid
        print('| Validation loss %.4f\tTop1error %.4f \tTop3error: %.4f' \
              % (epoch_loss, top1error, top3error))

        ################### save model based on best top3 error
        if top3error < best_top3:
            print('Saving model')
            best_top3 = top3error
            best_model = copy.deepcopy(model)
            state = {
                'model': best_model,
                'top3': best_top3,
                'args': args
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point + saved_model_fn + '.t7')
            print('=======================================================================')
            print('model saved to %s' % (save_point + saved_model_fn + '.t7'))

writer.close()
