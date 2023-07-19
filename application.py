#!/usr/bin/env python
# coding=utf-8

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

from log import logger
from device import calc_trans_duration
from device import DeviceType, ue_connect_bs


class JobStatus(Enum):
    Waiting = 1
    Running = 2
    Completed = 3


class TaskStatus(Enum):
    Waiting = 1
    Ready = 2
    Running = 3
    Completed = 4


@dataclass(frozen=True)
class Category():
    categoryId: int
    requestCPU: int
    requestMemory: int
    prepareDuration: int


@dataclass
class Task():
    taskId: int
    jobId: int
    categoryId: int
    parentTasks: List
    childTasks: List
    computeDuration: int = -1
    # 输出数据量大小，如果有多个出度，则取出度上最大的datasize
    outputsize: int = 0
    arriveTime: int = -1
    readyTime: int = -1
    startTime: int = -1
    endTime: int = -1
    status: str = TaskStatus.Waiting.name
    # 该Task在哪个Host/Executor上执行
    devicetype: str = ''
    deviceid: int = -1
    # 如果task在UE上执行，则数据需要save到关联的UE上
    savedevicetype: str = ''
    savedeviceid: int = -1

    executorid: int = -1
    real_compute_duration = -1

    def start_task(self, executorid, executor_matchtime, devicetype, deviceid, compute_factor):
        self.startTime = executor_matchtime
        self.real_compute_duration = self.computeDuration * compute_factor
        self.status = TaskStatus.Running.name
        self.executorid = executorid
        self.devicetype = devicetype
        self.deviceid = deviceid

    def end_task(self, expect_end_time):
        self.status = TaskStatus.Completed.name
        self.endTime = expect_end_time

    # Executor删除导致Task执行失败
    def fail_task(self):
        self.status = TaskStatus.Ready.name
        self.endTime = -1
        self.executorid = -1
        self.devicetype = ''
        self.deviceid = -1
        self.savedevicetype = ''
        self.savedeviceid = -1

    def get_expect_end_time(self, pt_param, cloud_df, host_df, bs_df, ue_metric_df):
        # 所有parenttask已完成的情况下，计算当前task的expect_end_time
        compute_start_time = self.startTime
        if len(self.parentTasks) > 0:
            for i, p in enumerate(self.parentTasks):
                trans_size = p[1]
                trans_starttime = max(self.startTime, pt_param[i].get('endtime'))
                src_devicetype = pt_param[i].get('devicetype')
                src_deviceid = pt_param[i].get('deviceid')
                trans_duration_p = calc_trans_duration(src_devicetype, src_deviceid, self.devicetype, self.deviceid,
                                                       trans_starttime, trans_size, cloud_df, host_df, bs_df, ue_metric_df)
                trans_time = max(self.startTime, pt_param[i].get('endtime')) + trans_duration_p
                compute_start_time = max(compute_start_time, trans_time)
        expect_end_time = compute_start_time + self.real_compute_duration
        # 如果该task在UE上执行，则数据需要写入到邻近的BS
        if self.devicetype == DeviceType.UE.name:
            bsid, trans_duration = ue_connect_bs(self.deviceid, expect_end_time, self.outputsize, ue_metric_df)
            # 传输过程中UE下线
            if trans_duration == -1:
                # self.fail_task()
                return -1
            expect_end_time += trans_duration
            self.savedevicetype = DeviceType.BS.name
            self.savedeviceid = bsid
        else:
            self.savedevicetype = self.devicetype
            self.savedeviceid = self.deviceid
        return expect_end_time


@dataclass
class Job():
    jobId: int
    arriveTime: int
    tasks: List[Task]
    startTime: int = -1
    endTime: int = -1
    status: str = JobStatus.Waiting.name

    def __post_init__(self):
        self.taskids = [i.taskId for i in self.tasks]
        for task_obj in self.tasks:
            task_obj.arriveTime = self.arriveTime
        for task_obj in filter(lambda x: len(x.parentTasks) == 0, self.tasks):
            task_obj.status = TaskStatus.Ready.name
            task_obj.readyTime = self.startTime
        self.endTime = self.arriveTime + 10000

    # Executor开始执行1个Task
    def start_one_task(self, executor_matchtime, task_id):
        if self.startTime == -1:
            self.startTime = executor_matchtime
        logger.debug('start a task.jobid:{},taskid:{}'.format(self.jobId, task_id))
        self.status = JobStatus.Running.name

    # 有1个Task执行成功
    def end_one_task(self, task_id, task_endtime):
        logger.debug('complete a task.jobid:{},taskid:{}'.format(self.jobId, task_id))
        # 该Task的所有子孙Taskids
        s_taskids = [ct[0] for ct in self.tasks[self.taskids.index(task_id)].childTasks]
        for s_task in filter(lambda x: (x.taskId in s_taskids) and (x.status == TaskStatus.Waiting.name),
                             self.tasks):
            s_task.status = TaskStatus.Ready.name
            s_task.readyTime = task_endtime
        if all([t.status == TaskStatus.Completed.name for t in self.tasks]):
            self.status = JobStatus.Completed.name
            self.endTime = max([t.endTime for t in self.tasks])
            logger.debug('completed job:{}'.format(self.jobId))

    # 有1个Task执行失败
    def fail_one_task(self, taskid):
        logger.debug('fail a task.jobid:{},taskid:{}'.format(self.jobId, taskid))

    @property
    def jct(self):
        return self.endTime - self.arriveTime
