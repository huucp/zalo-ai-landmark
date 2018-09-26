from __future__ import division
import argparse
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize

from zalo_utils import *
import numpy as np

TOTAL_CLASSES = 103
KTOP = 3


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def predict_multi_crop(inputs, predict_model):
    bs, n_crops, c, h, w = inputs.size()
    inputs = cvt_to_gpu(inputs)
    outputs = predict_model(inputs.view(-1, c, h, w))

    if isinstance(outputs, tuple):
        outputs, _ = outputs

    outputs = outputs.view(bs, n_crops, -1).mean(1)
    outputs = outputs.data.cpu().numpy()
    if outputs.ndim == 1:
        outputs = outputs.reshape(int(len(outputs) / TOTAL_CLASSES),
                                  int(TOTAL_CLASSES))  # fix when outputs has only 1 image
    return outputs


def gen_pred_row(fn, label, pred_array):
    idx = fn.split('/')[-1][:-4]
    return idx + ',' + str(label) + ',' + str(pred_array)[1:-1].replace(' ', '') + '\n'


def prediction(images_dir, truth_label, pred_model, model_name, batch_size, num_workers, file_writer):
    global fns0
    tot = 0
    fn_all = [images_dir + fn for fn in os.listdir(images_dir) if fn.endswith('.jpg')]
    fns = []
    fn_corrupted = []
    for fn in fn_all:
        # filter dammaged images
        if os.path.getsize(fn) > 0:
            fns.append(fn)
        else:
            fn_corrupted.append(fn)
    print('Total provided files: {}'.format(len(fn_all)))
    print('Total damaged files: {}'.format(len(fn_all) - len(fns)))

    lbs = [-1] * len(fns)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if 'inception' in model_name or 'xception' in model_name:
        scale_size = 333
        input_size = 299
    else:
        scale_size = 256
        input_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),  # simple data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            # transforms.Scale(scale_size),
            # transforms.CenterCrop(input_size),
            # transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            transforms.Resize(scale_size),
            transforms.FiveCrop(input_size),
            transforms.Lambda(
                lambda crops: torch.stack([Compose([ToTensor(), Normalize(mean, std)])(crop) for crop in crops]))
        ]),
    }

    dsets = dict()
    dsets['test'] = LandmarkDataSet(fns, lbs, transform=data_transforms['val'])

    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x],
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)
        for x in ['test']
    }

    error_count = 0
    total_count = 1

    for batch_idx, (inputs, labels, fns0) in enumerate(dset_loaders['test']):
        result = predict_multi_crop(inputs, pred_model)
        assert len(result) == len(fns0)
        outputs_ktop = np.argsort(result, axis=1)[:, -KTOP:][:, ::-1]
        for i in range(len(fns0)):
            file_writer.write(gen_pred_row(fns0[i], truth_label, list(result[i])))

            total_count += 1
            if truth_label not in list(outputs_ktop[i]):
                error_count += 1
        tot += len(fns0)
        print('processed {}/{}'.format(tot, len(fns)))
    print("error for class ", truth_label, ": ", "{0:.2f}".format(error_count * 100.0 / total_count))
    return error_count, total_count


def write(model_name, model_folder, batch_size=32, num_class=TOTAL_CLASSES,
          test_folder='../data/Public/', public_test=True, num_workers=2):
    pred_model, _ = load_model(model_folder + model_name)
    print("Write prediction for model:", model_name)
    pred_model.eval()
    file_writer = open("./predict_data/" + model_name.split('.')[0] + '-' + test_folder.split('/')[2] + ".dat",
                       "w")

    torch.set_grad_enabled(False)

    if public_test:
        data_dir = test_folder
        prediction(data_dir, -1, pred_model, model_name, batch_size, num_workers, file_writer)
    else:
        total_img = 1
        error_img = 0
        for idx in range(num_class):
            print("Predict for class: ", idx)
            data_dir = test_folder + str(idx) + "/"
            class_error, class_total = prediction(data_dir, idx, pred_model, model_name, batch_size, num_workers,
                                                  file_writer)
            print("error: ", class_error, "/", class_total)
            total_img += class_total
            error_img += class_error

        correct_img = total_img - error_img
        print("error percentage: ", "{0:.2f}".format(error_img * 100.0 / total_img))
        print("accuracy percentage: ", "{0:.2f}".format(correct_img * 100.0 / total_img))

    file_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zalo Landmark Identification Error Analysis')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_folder', type=str, default='./final_models/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_class', type=int, default=TOTAL_CLASSES)
    parser.add_argument('--test_folder', type=str, default='../data/Public/')
    parser.add_argument('--public_test', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    write(args.model_name, args.model_folder, args.batch_size, args.num_class, args.test_folder, args.public_test,
          args.num_workers)
