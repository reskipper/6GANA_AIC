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

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List
import math

import pandas as pd
from log import logger


class ExecutorStatus(Enum):
    Preparing = 1
    Busy = 2
    Free = 3


class DeviceType(Enum):
    Cloud = 1
    BS = 2
    UE = 3


@dataclass(order=True)
class Executor():
    sort_index: int = field(init=False, repr=False)
    executorId: int
    createTime: int
    categoryId: int
    requestCPU: int
    requestMemory: int
    prepareDuration: int
    status: str = ExecutorStatus.Preparing.name
    # 正在执行的taskid
    taskid: int = -1
    deleteTime: int = -1

    # 按照创建时间排序
    def __post_init__(self):
        self.sort_index = self.createTime

    def exec_task(self, taskid):
        self.taskid = taskid
        self.status = ExecutorStatus.Busy.name

    def end_task(self):
        self.taskid = -1
        self.status = ExecutorStatus.Free.name


@dataclass
class BaseHost():
    executors: List[Executor] = field(default_factory=list)
    computerFactor: int = -1
    rate: int = -1
    # 资源容量
    cpuCapacity: int = -1
    memoryCapacity: int = -1
    # 资源余量
    cpuMargin: int = -1
    memoryMargin: int = -1

    def create_exector(self, executor_obj):
        if (self.cpuMargin >= executor_obj.requestCPU) and (self.memoryMargin >= executor_obj.requestMemory):
            self.cpuMargin -= executor_obj.requestCPU
            self.memoryMargin -= executor_obj.requestMemory
            self.executors.append(executor_obj)
            return True
        else:
            return False

    def delete_executor(self, executor_obj):
        self.cpuMargin += executor_obj.requestCPU
        self.memoryMargin += executor_obj.requestMemory
        self.executors.remove(executor_obj)
        logger.debug('delete executor,executorid:{},executor status:{},executor run taskid:{}'. \
                     format(executor_obj.executorId, executor_obj.status, executor_obj.taskid))
        return True

    @property
    def executorids(self):
        return [i.executorId for i in self.executors]

    # CPU\Memory容量发生变更
    def change_capacity(self, cpu_capacity, memory_capacity):
        self.executors.sort(reverse=False)
        # 当前Host上已经占用的CPU\Memory
        cost_cpu = self.cpuCapacity - self.cpuMargin
        cost_memory = self.memoryCapacity - self.memoryMargin
        self.cpuCapacity = cpu_capacity
        self.memoryCapacity = memory_capacity
        # 由于删除Executors导致的Task执行失败
        failed_taskids = []
        # 删除超载的Executors
        while ((self.cpuCapacity < cost_cpu) or (self.memoryCapacity < cost_memory)) \
                and (len(self.executors) > 0):
            temp_executor = self.executors[0]
            self.delete_executor(temp_executor)
            cost_cpu -= temp_executor.requestCPU
            cost_memory -= temp_executor.requestMemory
            if temp_executor.status == ExecutorStatus.Busy.name:
                failed_taskids.append(temp_executor.taskid)
        # 用新的Capacity更新Margin
        self.cpuMargin = self.cpuCapacity - cost_cpu
        self.memoryMargin = self.memoryCapacity - cost_memory
        return failed_taskids


@dataclass
class Host(BaseHost):
    hostId: int = -1
    cloudId: int = -1


@dataclass
class BS(BaseHost):
    bsId: int = -1


@dataclass
class UE(BaseHost):
    ueId: int = -1
    bsId: int = -1
    onlineTime: int = -1
    offlineTime: int = -1

    def change_connection(self, bsid, rate):
        self.bsId = bsid
        self.rate = rate

    # ue下线,executor删除,task执行失败
    def offline(self):
        failed_taskids = []
        while len(self.executors) > 0:
            temp_executor = self.executors[0]
            self.delete_executor(temp_executor)
            if temp_executor.status == ExecutorStatus.Busy.name:
                failed_taskids.append(temp_executor.taskid)
        return failed_taskids


# 计算数据传输时延
def calc_trans_duration(src_devicetype, src_deviceid, tgt_devicetype, tgt_deviceid, trans_starttime, trans_size,
                        cloud_df, host_df, bs_df, ue_metric_df):
    trans_duration = 0
    if src_devicetype == tgt_devicetype and src_deviceid == tgt_deviceid:
        return trans_duration
    devicetype_s = {src_devicetype, tgt_devicetype}
    # Cloud-Cloud
    if len(devicetype_s ^ {DeviceType.Cloud.name}) == 0:
        if host_df.loc[src_deviceid, 'CloudId'] != host_df.loc[tgt_deviceid, 'CloudId']:
            trans_duration = math.ceil(trans_size / int(min(
                cloud_df.loc[
                    [host_df.loc[src_deviceid, 'CloudId'], host_df.loc[tgt_deviceid, 'CloudId']], 'Rate'].values)))
        return trans_duration
    # Cloud-BS
    if len(devicetype_s ^ {DeviceType.Cloud.name, DeviceType.BS.name}) == 0:
        if src_devicetype == DeviceType.BS.name:
            bs_rate = int(bs_df.loc[src_deviceid, 'Rate'])
        else:
            bs_rate = int(bs_df.loc[tgt_deviceid, 'Rate'])
        trans_duration = math.ceil(trans_size / bs_rate)
        return trans_duration
    # Cloud-UE
    if len(devicetype_s ^ {DeviceType.Cloud.name, DeviceType.UE.name}) == 0:
        if src_devicetype == DeviceType.UE.name:
            ue_id = src_deviceid
        else:
            ue_id = tgt_deviceid
        try:
            while trans_size > 0:
                trans_size -= int(ue_metric_df.loc[(ue_id, trans_starttime % 1200), 'Rate'])
                trans_starttime += 1
                trans_duration += 1
        # 传输过程中UE下线
        except KeyError as e:
            trans_duration = -1
        return trans_duration
    # BS-BS
    if len(devicetype_s ^ {DeviceType.BS.name}) == 0:
        bs_rate = int(min(bs_df.loc[[src_deviceid, tgt_deviceid], 'Rate'].values))
        trans_duration = math.ceil(trans_size / bs_rate)
        return trans_duration
    # BS-UE
    if len(devicetype_s ^ {DeviceType.BS.name, DeviceType.UE.name}) == 0:
        if src_devicetype == DeviceType.UE.name:
            ue_id = src_deviceid
        else:
            ue_id = tgt_deviceid
        try:
            while trans_size > 0:
                trans_size -= int(ue_metric_df.loc[(ue_id, trans_starttime % 1200), 'Rate'])
                trans_starttime += 1
                trans_duration += 1
        # 传输过程中UE下线
        except KeyError as e:
            trans_duration = -1
        return trans_duration
    # UE-UE
    if len(devicetype_s ^ {DeviceType.UE.name}) == 0:
        try:
            while trans_size > 0:
                trans_size -= int(
                    min(ue_metric_df.loc[[(src_deviceid, trans_starttime % 1200),
                                   (tgt_deviceid, trans_starttime % 1200)], 'Rate'].values))
                trans_starttime += 1
                trans_duration += 1
        # 传输过程中UE下线
        except KeyError as e:
            trans_duration = -1
        return trans_duration


# T时刻UE向邻近的BS传输datasize的数据。返回bsid和传输时长
def ue_connect_bs(ueid, trans_starttime, trans_size, ue_metric_df):
    trans_duration = 0
    try:
        bsid = ue_metric_df.loc[(ueid, trans_starttime % 1200), 'BSId']
        while trans_size > 0:
            trans_size -= int(ue_metric_df.loc[(ueid, trans_starttime % 1200), 'Rate'])
            trans_starttime += 1
            trans_duration += 1
    except KeyError as e:
        bsid = -1
        trans_duration = -1
    return bsid, trans_duration
