# 输出result

from device import DeviceType
from enum import Enum
import pandas as pd
import os


class ExecutorAction(Enum):
    Add = 1
    Delete = 2


class Writer:
    """
    示例
    writer = Writer('test0')
    writer.add_executor(1, 1, 1, 1)  即 time=1，category_id=1，device_type=Cloud，device_id=1
    writer.delete_executor(1, 1)  即 time=1，executor_id=1
    writer.add_task(1, 1, 1)  即 time=1，executor_id=1，task_id=1
    writer.write()
    """

    def __init__(self, dir):
        self.dir = './result/' + dir
        self.current_executor_id = -1
        self.executor_df = pd.DataFrame(columns=['Time', 'Action', 'ExecutorId', 'CategoryId', 'DeviceType', 'DeviceId'])
        self.task_df = pd.DataFrame(columns=['Time', 'ExecutorId', 'TaskId'])

    def add_executor_w(self, time, category_id, device_type, device_id):
        self.current_executor_id += 1
        
        new_row = pd.DataFrame({'Time': time,
                                'Action': ExecutorAction.Add.name, 'ExecutorId': self.current_executor_id, 'CategoryId': category_id,
                                'DeviceType': DeviceType(device_type).name, 'DeviceId': device_id}, index=[0])
        
        self.executor_df = pd.concat([self.executor_df, new_row]).reset_index(drop=True)

        # return self.current_executor_id
    
    def delete_executor_w(self, time, executor_id):        
        new_row = pd.DataFrame({'Time': time,
                                'Action': ExecutorAction.Delete.name, 'ExecutorId': executor_id,
                                'CategoryId': None,
                                'DeviceType': None,
                                'DeviceId': None}, index=[0])
        
        self.executor_df = pd.concat([self.executor_df, new_row]).reset_index(drop=True)

    def add_task_w(self, time, executor_id, task_id):
        new_row = pd.DataFrame({'Time': time,
                                'ExecutorId': executor_id,
                                'TaskId': task_id}, index=[0])
        
        self.task_df = pd.concat([self.task_df, new_row]).reset_index(drop=True)
    
    def write(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.executor_df.to_csv(self.dir + '/executor.csv', index=False)
        self.task_df.to_csv(self.dir + '/task.csv', index=False)
