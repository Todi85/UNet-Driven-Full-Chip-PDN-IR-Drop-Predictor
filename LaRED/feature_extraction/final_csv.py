import os
import csv

feature_path = './training_set/sample/feature'
out_test_csv = './test.csv'

if __name__ == '__main__':

    features = os.listdir(feature_path)
    features = [v for v in features]

    with open(out_test_csv, 'w') as f_test:

        f_test_csv = csv.writer(f_test, delimiter=',')

        for i, features_name in enumerate(features):
            features_path = 'training_set/sample/feature/{}'.format(features_name)
            # 不用管，最后test.py里如果finaltest设置为true会在那里进行替换
            instance_count_path = 'out/features/instance_count/{}'.format(features_name)
            instance_name_path = 'out/features/instance_name/{}z'.format(features_name[:-1])

            f_test_csv.writerow([features_path, instance_count_path, instance_name_path])



