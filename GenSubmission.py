import WriteModelPrediction
import os
import AverageEnsemble

test_folder = './private_test_3_9/'
model_dir = './final_models/'
predict_dir = './predict_data/'

# delete all file in predict dir
for the_file in os.listdir(predict_dir):
    file_path = os.path.join(predict_dir, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

all_model = [fn for fn in os.listdir(model_dir) if fn.endswith('.t7')]

for model in all_model:
    WriteModelPrediction.write(model, model_dir, 32, 103, test_folder, True)

all_predict = [predict_dir + fn for fn in os.listdir(predict_dir) if fn.endswith('.dat')]

AverageEnsemble.ensemble(all_predict, True, False, None)
