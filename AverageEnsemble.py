from __future__ import division

from zalo_utils import *

TOTAL_CLASSES = 103
KTOP = 3


def gen_outputline(fn, preds):
    return str(int(fn)) + ',' + str(preds)[1:-1].replace(',', '') + '\n'


def gen_error_line(fn, preds, results, label):
    return str(int(fn)) + ',' + str(label) + ',' + str(preds)[1:-1].replace(' ', '') + ',' + str(results)[1:-1].replace(
        ' ',
        '') + '\n'


def ensemble(files, write_submission, error_analysis, file_name):
    pred_files = []
    for file in files:
        data = np.loadtxt(file, delimiter=',')
        pred_files.append(data)

    mean = np.mean(pred_files, axis=0)
    result = []
    for r in mean:
        row = np.array([])
        row = np.append(row, r[2:])
        result.append(row)

    pred_label = np.argsort(result, axis=1)[:, -KTOP:][:, ::-1]
    error_count = 0
    pred_size = len(pred_label)
    for idx in range(pred_size):
        if pred_files[0][idx][1] not in pred_label[idx]:
            error_count += 1

    print("Error:", error_count, "/", pred_size)
    print("Error percentage: ", "{0:.2f}".format(error_count * 100.0 / pred_size))
    # print file
    if write_submission:
        res_fn = './result/submission.csv'
        with open(res_fn, 'w') as f:
            header = 'id,predicted\n'
            f.write(header)
            for idx in range(pred_size):
                f.write(gen_outputline(pred_files[0][idx][0], list(pred_label[idx])))

    if error_analysis:
        error_writer = open("error-" + file_name + ".csv", 'w')
        correct_writer = open("correct-" + file_name + ".csv", 'w')

        for idx in range(pred_size):
            soft_result = softmax(result[idx])
            truth_label = pred_files[0][idx][1]
            if truth_label not in pred_label[idx]:
                error_writer.write(
                    gen_error_line(pred_files[0][idx][0], list(pred_label[idx]), list(soft_result), truth_label))
            else:
                correct_writer.write(
                    gen_error_line(pred_files[0][idx][0], list(pred_label[idx]), list(soft_result), truth_label))
        error_writer.close()
        correct_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zalo Landmark Identification Error Analysis')
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--write_submission', type=str2bool, default=False)
    parser.add_argument('--error_analysis', type=str2bool, default=False)
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    ensemble(args.files, args.write_submission, args.error_analysis, args.file_name)
