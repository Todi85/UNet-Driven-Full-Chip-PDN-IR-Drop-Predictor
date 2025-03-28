import os, gzip
import json
import numpy as np
import torch
from tqdm import tqdm
import time
from datasets.build_dataset import build_dataset
import utils.metrics as metrics
from models.build_model import build_model
from utils.arg_parser import Parser
from utils.logger import build_logger

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import ndimage

def resize(input, out_shape):
    dimension = input.shape
    result = ndimage.zoom(input, (out_shape[0] / dimension[0], out_shape[1] / dimension[1]), order=3)
    return result

def build_metric(metric_name):
    return metrics.__dict__[metric_name.lower()]

def test():
    
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    gpu = None
    pretrained = None
    if arg.gpu is not None:
        gpu = int(arg.gpu)
    if arg.pretrained is not None:
        pretrained = arg.pretrained


    if arg.args is not None:
        with open(arg.args, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['test_mode'] = True


    if gpu is not None:
        arg_dict['gpu'] = gpu
    if pretrained is not None:
        arg_dict['pretrained'] = pretrained
    if pretrained is not None and arg_dict['test_mode']:
        arg_dict['save_path'] = os.path.dirname(pretrained)


    logger, log_dir = build_logger(arg_dict)
    logger.info(arg_dict)
    if not os.path.exists(os.path.join(log_dir, 'pred_static_ir_report')):
        os.makedirs(os.path.join(log_dir, 'pred_static_ir_report'))
        
    if arg_dict['cpu']:
        device = torch.device("cpu")
        logger.info('using cpu for training')
    elif arg_dict['gpu'] is not None:
        torch.cuda.set_device(arg_dict['gpu'])
        device = torch.device("cuda", arg_dict['gpu'])
        logger.info('using gpu {} for training'.format(arg_dict['gpu']))
        
    logger.info('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    logger.info('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.to(device)

    # Build metrics
    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}
    split_metrics = {k:{} for k in arg_dict['eval_metric']}

    count = 1
    start = True
    #if not arg.final_test:
    for feature, label, instance_count_path, instance_IR_drop_path, instance_name_path in dataset:
            design_name = os.path.basename(instance_count_path[0])
            if 'FPU' in design_name:
                design_name = 'RISCY-FPU'
            else:
                design_name = design_name.split('_')[0]
            if design_name not in list(split_metrics.values())[0].keys() or start:
                for i in split_metrics.keys():
                    split_metrics[i][design_name] = [0, 0]
                start = False
            if arg_dict['cpu']:
                input = feature
                start_time = time.time()
                prediction = model(input)
                end_time = time.time()

            else:
                input = feature.to(device)
                # print(input.size())
                torch.cuda.synchronize()
                start_time = time.time()
                prediction = model(input)
                torch.cuda.synchronize()
                end_time = time.time()
            if not arg.final_test:
                logger.info('#{} {}, inference time {}s'.format(count, os.path.basename(instance_count_path[0][:-4]),
                                                            end_time - start_time))
            if not arg.final_test:
                instance_count = np.load(os.path.join(instance_count_path[0])).astype(int)
                instance_name = np.load(instance_name_path[0])['instance_name']  # load npz

            else:  # final test, get instance name from power rpt
                instance_count = np.load(instance_count_path[0].replace('instance_count', 'instance_count_from_power_rpt')).astype(int)
                instance_name = np.load(instance_name_path[0].replace('instance_name', 'instance_name_from_power_rpt'))['instance_name']  # load npz

            pred_vdd_gnd = resize(prediction[0, 0, :, :].detach().cpu().numpy(), instance_count.shape)
            pred_instance_vdd_gnd = np.repeat(pred_vdd_gnd.ravel(), instance_count.ravel())
            # 输出预测的static_ir report
            # 文件为2列，第一列是vdd_drop+gnd_bounce，第二列是inst_name，不需要表头。
            file_name = os.path.splitext(os.path.basename(instance_count_path[0]))[0]
            with open('{}/{}/{}'.format(log_dir, 'pred_static_ir_report', 'pred_static_ir_{}'.format(file_name)),
                      'w') as f:
                for j, k in zip(pred_instance_vdd_gnd, instance_name):
                    f.write('{} {} \n'.format(j, k))

            if not arg.final_test:
                # 打印预测性能
                instance_IR_drop = np.load(instance_IR_drop_path[0])
                for metric, metric_func in metrics.items():
                    # result = metric_func(instance_IR_drop, pred_instance_vdd_drop + pred_instance_gnd_bounce)
                    result = metric_func(instance_IR_drop, pred_instance_vdd_gnd)
                    logger.info('{}: {}'.format(metric, result))
                    avg_metrics[metric] += result
                    split_metrics[metric][design_name][0] = split_metrics[metric][design_name][0] + result
                    split_metrics[metric][design_name][1] = split_metrics[metric][design_name][1] + 1

            if arg_dict['plot']:
                save_path = os.path.join(log_dir, 'test_result_visualization')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                output_final = prediction.detach().cpu().squeeze().numpy() * 1000
                label = label * 1000

                fig = sns.heatmap(data=output_final, cmap="hot", vmin=0, vmax=20).get_figure()
                fig.savefig(os.path.join(save_path, file_name + '_pred_vdd_gnd.png'), dpi=100)
                plt.close()

                fig = sns.heatmap(data=label.squeeze().numpy(), cmap="hot", vmin=0, vmax=20).get_figure()
                fig.savefig(os.path.join(save_path, file_name + '_label_vdd_gnd.png'), dpi=100)
                plt.close()

            # if count==10:
            #     break
            count += 1
    if not arg.final_test:
        for metric, avg_metric in avg_metrics.items():
            logger.info("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))
        for metric, design in split_metrics.items():
            if len(design) == 1:
                continue
            for name, values in design.items():
                logger.info("===> {} {}: {:.4f}".format(name, metric, values[0] / values[1]))

    logger.info("Predicted static_ir report saved in {}/{}.".format(log_dir, 'pred_static_ir_report')) 


if __name__ == "__main__":
    test()
