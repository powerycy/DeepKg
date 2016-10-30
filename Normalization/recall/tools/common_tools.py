# encoding utf8
# author : King
# time : 2020.08.31
# decs ： 通用 工具
''' 函数目录

    1. timer 计数器


'''
import time
# 功能：装饰器 之 计数器
def timer(func):
    '''
        功能：装饰器 之 计数器
        操作：在 需要计算的函数 前面 加上 @timer
    '''
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print('函数：{}() 共耗时约 {:.5f} 秒'.format(func.__name__,time.time() - start))
        return res
    return wrapper