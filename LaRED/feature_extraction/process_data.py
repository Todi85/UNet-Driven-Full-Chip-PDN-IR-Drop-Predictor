import os
from multiprocessing import Process
import argparse
from src.util import divide_n
from src.read import Paraser


class ArgParaser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_root', default='./data', help='the parent dir of log dirs')
        self.parser.add_argument('--save_path', default='./out', help='save path')
        #这里把默认改了
        self.parser.add_argument('--process_capacity', default=8, help='number of process for multi process')
        self.parser.add_argument('--plot', default=False, help='plot the results in $save_path/visual')
        self.parser.add_argument('--debug', default=False, help='disable multi process to use pdb')
        self.parser.add_argument('--final_test', default=False, help='prevent using static_ir to mimic the final environment')


'''def read(read_list,save_list, arg):

    for path,path_save in zip(read_list,save_list):
        save_name = path_save
        path = os.path.join(arg.data_root, path)
        process_log = Paraser(path, arg, save_name)
        print(save_name)
        process_log.get_IR_drop_features()     '''
def read(read_list, arg):

    for path in read_list:
        save_name = path
        path = os.path.join(arg.data_root, path)
        process_log = Paraser(path, arg, save_name)
        print(save_name)
        process_log.get_IR_drop_features()


if __name__ == '__main__':
    argp = ArgParaser()
    arg = argp.parser.parse_args()
    if not os.path.exists(arg.save_path):
        os.makedirs(arg.save_path)
    read_list = os.listdir(arg.data_root)
    nlist = divide_n(read_list, arg.process_capacity)

    if arg.debug:
        read(read_list, arg)
    else:
        process = []
        for divided_list in nlist:
            p = Process(target=read, args=(divided_list, arg))
            process.append(p)
        for p in process:
            p.start()
        for p in process:
            p.join()
    '''read_list = os.listdir(arg.data_root)
    #二级目录
    full_list = []
    save_list=[]
    for i in read_list:
        i_list = os.listdir(os.path.join(arg.data_root, i))
        for j in i_list:
            #print(os.path.join(i, j))
            full_list.append(os.path.join(i, j))
            save_list.append(j)

    #nlist = divide_n(read_list, arg.process_capacity) #一级目录的处理方法
    nlist = divide_n(full_list, arg.process_capacity)
    nlist_save = divide_n(save_list, arg.process_capacity)
    if arg.debug:
        read(read_list, arg)
    else:
        process = []
        for divided_list,divided_list_save in zip(nlist,nlist_save):
            p = Process(target=read, args=(divided_list,divided_list_save, arg))
            process.append(p)
        for p in process:
            p.start()
        for p in process:
            p.join()'''