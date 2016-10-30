# encoding utf8
# author : King
# time : 2019.11.12
''' 函数目录

    1. class Multiprocessing_class(): 功能：多进程计算


'''
import multiprocessing
class Multiprocessing_class():
    def __init__(self,processes_num):
        self.processes_num = processes_num          # 进程数量
    
       # 功能：多进程模块
    def use_multiprocessing_for_list(self,fun,lists,block_size,*args):
        '''
        功能：多进程模块
        :param fun:                 fun                函数
        :param lists:               List               句子列表
        :param block_size:          Int                分块大小
        :param *args:               根据任务选取不同的参数
        :return:
            labeled_sent_list       List[List]    句子列表  
        '''
        # 构建语料模块
        pool = multiprocessing.Pool(processes = self.processes_num)
        process = []                # 进程存储列表
        for i in range(self.processes_num):
            process.append(pool.apply_async(fun, args=(tuple([lists[block_size*i:(i+1)*block_size]]+list(args)))))

        print("~~~~~~~~~~multiprocessing start~~~~~~~~~~~~")
        pool.close()
        pool.join()    # behind close() or terminate()
        print("Sub-process(es) done.")

        res_list = []
        for i in process:
            res_list = res_list + i.get()
        return res_list



