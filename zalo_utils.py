from __future__ import print_function
import os

from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.models as models
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
from time import time
import xception

def load_model(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model = unparallelize_model(model)
    model = parallelize_model(model)
    best_top3 = checkpoint['top3']
    return model, best_top3


def get_fns_lbs(base_dir, json_file, pickle_fn='mydata.p', force=False):
    pickle_fn = base_dir + pickle_fn
    # pdb.set_trace() 
    if os.path.isfile(pickle_fn) and not force:
        mydata = pickle.load(open(pickle_fn, 'rb'))
        fns = mydata['fns']
        lbs = mydata['lbs']
        cnt = mydata['cnt']
        return fns, lbs, cnt

    f = open(json_file, 'r')
    line = f.readlines()[0]  # only one line
    end = 0
    id_marker = '\\"id\\": '
    cate_marker = '\\"category\\": '
    cnt = 0
    fns = []  # list of all image filenames
    lbs = []  # list of all labels
    while True:
        start0 = line.find(id_marker, end)
        if start0 == -1: break
        start_id = start0 + len(id_marker)
        end_id = line.find(',', start_id)

        start0 = line.find(cate_marker, end_id)
        start_cate = start0 + len(cate_marker)
        end_cate = line.find('}', start_cate)

        end = end_cate
        cnt += 1
        cl = line[start_cate:end_cate]
        fn = base_dir + cl + '/' + line[start_id:end_id] + '.jpg'
        if os.path.getsize(fn) == 0:  # zero-byte files
            continue
        lbs.append(int(cl))
        fns.append(fn)

    # pdb.set_trace()
    mydata = {'fns': fns, 'lbs': lbs, 'cnt': cnt}
    pickle.dump(mydata, open(pickle_fn, 'wb'))
    print(os.path.isfile(pickle_fn))

    return fns, lbs, cnt


class LandmarkDataSet(Dataset):

    def __init__(self, filenames, labels, transform=None):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.transform = transform

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        # TODO: replace Image by opencv
        image = Image.open(self.fns[idx])
        tmp = image.getpixel((0, 0))
        if isinstance(tmp, int) or len(tmp) != 3:  # not rgb image
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.lbs[idx], self.fns[idx]


class MyDenseNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained=True):
        super(MyDenseNet, self).__init__()
        if depth == 121:
            self.model = models.densenet121(pretrained)
        elif depth == 161:
            self.model = models.densenet161(pretrained)
        elif depth == 169:
            self.model = models.densenet169(pretrained)
        elif depth == 201:
            self.model = models.densenet201(pretrained)

        num_ftrs = self.model.classifier.in_features
        # self.num_classes = num_classes
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # pdb.set_trace()

        return self.model.forward(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive' % to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.model.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class MyXception(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MyXception, self).__init__()

        self.model = xception.xception(pretrained=pretrained)
        print(self.model.parameters())

        # Last layer
        num_ftrs = 2048
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # pdb.set_trace()
        return self.model.forward(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive' % to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.model.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class MyInception(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MyInception, self).__init__()

        self.model = models.inception_v3(pretrained)
        # Auxiliary Fc layer
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Last layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # pdb.set_trace()
        return self.model.forward(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive' % to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.model.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class MyResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained=True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 101:
            model = models.resnet101(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive' % to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


def mytopk(pred, gt, k=3):
    """
    compute topk error
    pred: (n_sample,n_class) np array
    gt: a list of ground truth
    --------
    return:
        n_correct: number of correct prediction 
        topk_error: error, = n_connect/len(gt)
    """
    # topk = np.argpartition(pred, -k)[:, -k:]
    topk = np.argsort(pred, axis=1)[:, -k:][:, ::-1]
    diff = topk - np.array(gt).reshape((-1, 1))
    n_correct = np.where(diff == 0)[0].size
    topk_error = float(n_correct) / pred.shape[0]
    return n_correct, topk_error


def net_frozen(args, model):
    print('********************************************************')
    model.frozen_until(args.frozen_until)
    init_lr = args.lr
    if args.trainer.lower() == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=init_lr, weight_decay=args.weight_decay)
    elif args.trainer.lower() == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=init_lr, weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer


def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def second2str(second):
    h = int(second / 3600.)
    second -= h * 3600.
    m = int(second / 60.)
    s = int(second - m * 60)
    return "{:d}:{:02d}:{:02d} (s)".format(h, m, s)


def print_eta(t0, cur_iter, total_iter):
    """
    print estimated remaining time
    t0: beginning time
    cur_iter: current iteration
    total_iter: total iterations
    """
    time_so_far = time() - t0
    iter_done = cur_iter + 1
    iter_left = total_iter - cur_iter - 1
    second_left = time_so_far / float(iter_done) * iter_left
    s0 = 'Epoch: ' + str(cur_iter + 1) + '/' + str(total_iter) + ', time so far: ' \
         + second2str(time_so_far) + ', estimated time left: ' + second2str(second_left)
    print(s0)


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

# class MyInceptionV4(nn.Module):
#     def __init__(self, num_classes, pretrained=True):
#         super(MyInceptionV4, self).__init__()
#
#         self.model = pretrainedmodels.inceptionv4(pretrained='imagenet')
#
#         # Last layer
#         num_ftrs = self.model.last_linear.in_features
#         self.model.last_linear = nn.Linear(num_ftrs, num_classes)
#
#     def forward(self, x):
#         # pdb.set_trace()
#         return self.model.forward(x)
#
#     def frozen_until(self, to_layer):
#         print('Frozen shared part until %d-th layer, inclusive' % to_layer)
#
#         # if to_layer = -1, frozen all
#         child_counter = 0
#         for child in self.model.children():
#             if child_counter <= to_layer:
#                 print("child ", child_counter, " was frozen")
#                 for param in child.parameters():
#                     param.requires_grad = False
#                 # frozen deeper children? check
#                 # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
#             else:
#                 print("child ", child_counter, " was not frozen")
#                 for param in child.parameters():
#                     param.requires_grad = True
#             child_counter += 1

# class MyPolyNet(nn.Module):
#     def __init__(self, num_classes, pretrained=True):
#         super(MyPolyNet, self).__init__()
#         self.model = pretrainedmodels.polynet()
#         num_ftrs = self.model.last_linear.in_features
#         self.model.last_linear = nn.Linear(num_ftrs, num_classes)
#
#     def forward(self, x):
#         return self.model.forward(x)
#
#     def frozen_until(self, to_layer):
#         print('Frozen shared part until %d-th layer, inclusive' % to_layer)
#
#         # if to_layer = -1, frozen all
#         child_counter = 0
#         for child in self.model.children():
#             if child_counter <= to_layer:
#                 print("child ", child_counter, " was frozen")
#                 for param in child.parameters():
#                     param.requires_grad = False
#                 # frozen deeper children? check
#                 # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
#             else:
#                 print("child ", child_counter, " was not frozen")
#                 for param in child.parameters():
#                     param.requires_grad = True
#             child_counter += 1
