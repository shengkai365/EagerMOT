import os
from typing import Iterable, IO, Optional


def makedirs_if_new(path: str) -> bool:
    '''如果不存在path那么新建目录
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def close_files(files: Iterable[Optional[IO]]) -> None:
    '''关闭文件序列
    args:
        files: 文件描述符的可迭代对象
    '''
    for f in files:
        if f is not None:
            f.close()

def create_writable_file_if_new(folder: str, name: str):
    '''根据目录名和序列号创建并获取可写文件描述符
    Args:
        folder: 跟踪结果目录
        name: 数据集序列号
    Returns:
        如果存在 folder/xxxx.txt的文件, 返回None;
        否则返回 可写文件描述符
    '''
    makedirs_if_new(folder)
    results_file = os.path.join(folder, name + '.txt')
    if os.path.isfile(results_file):
        return None
    return open(results_file, 'w')
