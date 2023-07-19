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
import sys
import json
import itertools
import argparse

import pandas as pd
import numpy as np

from application import *
from device import *
from log import logger
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# close warning
pd.set_option('mode.chained_assignment', None)


def check_data(resultpath: str, datanames: List):
    """
    检查测试集结果文件是否都存在
    """
    valid = True
    if len(set(datanames) - set(os.listdir(resultpath))) > 0:
        valid = False
    result_files = {'executor.csv', 'task.csv'}
    for dn in datanames:
        if len(result_files - set(os.listdir(os.path.join(resultpath, dn)))) > 0:
            valid = False
            break
    if not valid:
        logger.warning('result path miss files')
        sys.exit(0)


class Environment(object):
    def __init__(self, data_path: str, plan_path: str):
        self.data_path = data_path
        self.plan_path = plan_path
        # 系统时间
        self.walltime = 0
        self.device_metric_duration = 1200
        self.bs_metric_period = 600

        self.load_taskcategory()
        self.load_init_device()
        self.load_init_jobs()

    def load_taskcategory(self):
        """
        taskcategory_df: key CategoryId, columns ['RequestCPU', 'RequestMemory', 'ComputeTime','PrepareTime']
        """
        taskcategory_df = pd.read_csv(os.path.join(self.data_path, 'category_table.csv'))
        taskcategory_j = json.loads(taskcategory_df.to_json(orient='records'))
        taskcategorys = [Category(*tc.values()) for tc in taskcategory_j]
        self.taskcategorys = taskcategorys

    def load_init_device(self):
        # 获取0时刻的Host状态
        # load cloud
        cloud_df = pd.read_csv(os.path.join(self.data_path, 'cloud_table.csv'))
        host_df = pd.read_csv(os.path.join(self.data_path, 'host_table.csv'))
        host_df1 = host_df.merge(cloud_df, on='CloudId')
        hosts = [
            Host(hostId=int(hostline.HostId), cloudId=int(hostline.CloudId), computerFactor=int(hostline.ComputeFactor),
                 rate=int(hostline.Rate), cpuCapacity=int(hostline.CPU),
                 memoryCapacity=int(hostline.Memory), cpuMargin=int(hostline.CPU), \
                 memoryMargin=int(hostline.Memory)) for hostline in host_df1.itertuples()]

        # load bs
        bs_df = pd.read_csv(os.path.join(self.data_path, 'bs_table.csv'))
        bs_metric_df = pd.read_csv(os.path.join(self.data_path, 'bs_metric.csv'))
        bs_metric_df.set_index(['BSId', 'Time'], inplace=True)
        bss = [BS(bsId=int(bsline.BSId), rate=int(bsline.Rate),
                  computerFactor=int(bsline.ComputeFactor),
                  cpuCapacity=int(bs_metric_df.loc[(bsline.BSId, 0), 'CPU']), \
                  memoryCapacity=int(bs_metric_df.loc[(bsline.BSId, 0), 'Memory']), \
                  cpuMargin=int(bs_metric_df.loc[(bsline.BSId, 0), 'CPU']), \
                  memoryMargin=int(bs_metric_df.loc[(bsline.BSId, 0), 'Memory']))
               for bsline in bs_df.itertuples()]
        ue_df = pd.read_csv(os.path.join(self.data_path, 'ue_table.csv'))
        ue_df.set_index('UEId', inplace=True)
        ue_metric_df = pd.read_csv(os.path.join(self.data_path, 'ue_metric.csv'))
        ue_metric_df.set_index('UEId', inplace=True, drop=False)
        ues = [UE(ueId=int(ueid), bsId=int(ue_metric_df.loc[ueid].iloc[0]['BSId']), \
                  onlineTime=int(ue_df.loc[ueid, 'OnlineTime']), \
                  offlineTime=int(ue_df.loc[ueid, 'OfflineTime']),
                  rate=int(ue_metric_df.loc[ueid].iloc[0]['Rate']), \
                  computerFactor=int(ue_df.loc[ueid, 'ComputeFactor']),
                  cpuCapacity=int(ue_metric_df.loc[ueid].iloc[0]['CPU']),
                  memoryCapacity=int(ue_metric_df.loc[ueid].iloc[0]['Memory']),
                  cpuMargin=int(ue_metric_df.loc[ueid].iloc[0]['CPU']),
                  memoryMargin=int(ue_metric_df.loc[ueid].iloc[0]['Memory'])) for
               ueid in ue_df.index.values]
        
        cloud_df.set_index('CloudId', inplace=True)
        host_df.set_index('HostId', inplace=True)
        bs_df.set_index('BSId', inplace=True)
        ue_metric_df.set_index(['UEId', 'Time'], inplace=True)
        self.cloud_df = cloud_df
        self.host_df = host_df
        self.bs_df = bs_df
        self.bs_metric_df = bs_metric_df
        self.ue_metric_df = ue_metric_df
        self.hosts = hosts
        self.bss = bss
        self.ues = ues

    def load_init_jobs(self):
        job_df = pd.read_csv(os.path.join(self.data_path, 'job_table.csv'))
        task_df = pd.read_csv(os.path.join(self.data_path, 'task_table.csv'))
        task_df.set_index('JobId', inplace=True)
        jobs = []
        task_n = 0
        for job_line in job_df.itertuples():
            job_tasks = []
            for task_line in task_df.loc[job_line.JobId].itertuples():
                task_output_size = 0
                if len(eval(task_line.ChildTasks)) > 0:
                    task_output_size = sum([ct[1] for ct in
                                            eval(task_line.ChildTasks)])
                job_task = Task(taskId=int(task_line.TaskId), jobId=int(job_line.JobId),
                                categoryId=int(task_line.CategoryId), \
                                parentTasks=eval(task_line.ParentTasks),
                                childTasks=eval(task_line.ChildTasks),
                                computeDuration=int(task_line.ComputeDuration), outputsize=task_output_size)
                job_tasks.append(job_task)
            task_n += len(job_tasks)

            job_obj = Job(jobId=int(job_line.JobId), arriveTime=int(job_line.ArriveTime), tasks=job_tasks)
            jobs.append(job_obj)
        self.jobs = jobs
        self.task_num = task_n

    def _get_ptask_param(self, jobid, taskid):
        # 返回该tasid的savedevicetype,savedeviceid,endtime
        task_obj = list(filter(lambda x: x.taskId == taskid, self.jobs[jobid].tasks))[0]
        return {'taskid': taskid, 'devicetype': task_obj.savedevicetype, 'deviceid': task_obj.savedeviceid,
                'endtime': task_obj.endTime}

    def _get_executor_obj(self, executor_id):
        # 找到Executor属于哪个Host
        for h in self.hosts:
            if executor_id in h.executorids:
                executor_obj = list(filter(lambda x: x.executorId == executor_id, h.executors))[0]
                executor_devicetype = DeviceType.Cloud.name
                executor_deviceid = h.hostId
                executor_computefactor = h.computerFactor
                return (executor_obj, executor_devicetype, executor_deviceid, executor_computefactor)

        for b in self.bss:
            if executor_id in b.executorids:
                executor_obj = list(filter(lambda x: x.executorId == executor_id, b.executors))[0]
                executor_devicetype = DeviceType.BS.name
                executor_deviceid = b.bsId
                executor_computefactor = b.computerFactor
                return (executor_obj, executor_devicetype, executor_deviceid, executor_computefactor)
        for u in self.ues:
            if executor_id in u.executorids:
                executor_obj = list(filter(lambda x: x.executorId == executor_id, u.executors))[0]
                executor_devicetype = DeviceType.UE.name
                executor_deviceid = u.ueId
                executor_computefactor = u.computerFactor
                return (executor_obj, executor_devicetype, executor_deviceid, executor_computefactor)
        return None

    def _get_task_obj(self, task_id):
        for job in self.jobs:
            if task_id in job.taskids:
                task_obj = list(filter(lambda x: x.taskId == task_id, job.tasks))[0]
                return task_obj
        return None

    def forward_walltime(self, walltime):
        before_walltime = self.walltime
        after_walltime = walltime
        logger.debug('forward walltime,from {} to {}.'.format(before_walltime, after_walltime))
        self.walltime = walltime
        # Task状态改变,Running变成Completed
        for job_obj in self.jobs:
            # 判断正在running的task，allparenttask已完成
            for task_obj in filter(lambda x: (x.status == TaskStatus.Running.name) and (
                    all([self._get_task_obj(p[0]).status == TaskStatus.Completed.name for p in
                         x.parentTasks])), job_obj.tasks):
                ptask_params = []
                if len(task_obj.parentTasks) > 0:
                    for pt in task_obj.parentTasks:
                        pt_taskid = pt[0]
                        ptask_params.append(self._get_ptask_param(task_obj.jobId, pt_taskid))
                task_expect_end_time = task_obj.get_expect_end_time(
                    ptask_params, self.cloud_df, self.host_df, self.bs_df, self.ue_metric_df)
                if task_expect_end_time > 0 and task_expect_end_time <= self.walltime:
                    task_obj.end_task(task_expect_end_time)
                    job_obj.end_one_task(task_obj.taskId, task_obj.endTime)
                    # 执行task的executor变成Free
                    if task_obj.devicetype == DeviceType.Cloud.name:
                        executor_obj = \
                            list(filter(lambda x: x.executorId == task_obj.executorid,
                                        self.hosts[task_obj.deviceid].executors))[
                                0]
                    elif task_obj.devicetype == DeviceType.BS.name:
                        executor_obj = \
                            list(filter(lambda x: x.executorId == task_obj.executorid,
                                        self.bss[task_obj.deviceid].executors))[0]
                    else:
                        executor_obj = \
                            list(filter(lambda x: x.executorId == task_obj.executorid,
                                        self.ues[task_obj.deviceid].executors))[0]
                    executor_obj.end_task()

        # Executor状态改变,Preparing变成Free
        all_executors = itertools.chain(*[h.executors for h in self.hosts], *[b.executors for b in self.bss],
                                        *[u.executors for u in self.ues])
        for executor_obj in filter(lambda x: (x.status == ExecutorStatus.Preparing.name) and (
                (x.createTime + x.prepareDuration) <= after_walltime), all_executors):
            executor_obj.status = ExecutorStatus.Free.name
        # Host资源容量改变
        # BS容量改变
        failed_taskids = []
        if (before_walltime // self.bs_metric_period) != (after_walltime // self.bs_metric_period):
            logger.debug('bs capacity changed')
            bs_metric_time_before = before_walltime // self.bs_metric_period * self.bs_metric_period
            bs_metric_time_after = after_walltime // self.bs_metric_period * self.bs_metric_period

            for bs_obj in self.bss:
                for bs_metric_time in range(bs_metric_time_before, bs_metric_time_after, self.bs_metric_period):
                    bs_metric_time_n = (bs_metric_time + self.bs_metric_period) % self.device_metric_duration
                    bs_cpu_capacity = int(self.bs_metric_df.loc[(bs_obj.bsId, bs_metric_time_n), 'CPU'])
                    bs_memory_capacity = int(self.bs_metric_df.loc[(bs_obj.bsId, bs_metric_time_n), 'Memory'])
                    bs_failed_taskids = bs_obj.change_capacity(bs_cpu_capacity, bs_memory_capacity)
                    failed_taskids.extend(bs_failed_taskids)
        # UE容量和连接改变
        if before_walltime != after_walltime:
            logger.debug('ue capacity and connection changed')
            for ue_metric_time in range(before_walltime, after_walltime, 1):
                ue_metric_time_n = (ue_metric_time + 1) % self.device_metric_duration
                online_ues = filter(
                    lambda x: (ue_metric_time_n >= x.onlineTime) and (ue_metric_time_n <= x.offlineTime),
                    self.ues)
                for online_ue in online_ues:
                    online_ueid = online_ue.ueId
                    ue_cpu_capacity = int(self.ue_metric_df.loc[(online_ueid, ue_metric_time_n), 'CPU'])
                    ue_memory_capacity = int(self.ue_metric_df.loc[(online_ueid, ue_metric_time_n), 'Memory'])
                    ue_faield_taskids = self.ues[online_ueid].change_capacity(ue_cpu_capacity, ue_memory_capacity)
                    failed_taskids.extend(ue_faield_taskids)
                offline_ues = filter(
                    lambda x: (x.offlineTime == ue_metric_time % self.device_metric_duration), self.ues)
                for offline_ue in offline_ues:
                    ue_faield_taskids = self.ues[offline_ue.ueId].offline()
                    failed_taskids.extend(ue_faield_taskids)
            # 更新UE的bs和rate
            online_ues = filter(
                lambda x: (after_walltime % self.device_metric_duration >= x.onlineTime) and (after_walltime % self.device_metric_duration <= x.offlineTime),
                self.ues)
            for online_ue in online_ues:
                online_ueid = online_ue.ueId
                ue_bs = int(self.ue_metric_df.loc[(online_ueid, after_walltime % self.device_metric_duration), 'BSId'])
                ue_rate = int(self.ue_metric_df.loc[(online_ueid, after_walltime % self.device_metric_duration), 'Rate'])
                self.ues[online_ueid].change_connection(ue_bs, ue_rate)
        if len(failed_taskids) > 0:
            logger.debug('host capacity change.failed taskids:{}'.format(failed_taskids))
            for fail_taskid in failed_taskids:
                fail_task_obj = self._get_task_obj(fail_taskid)
                fail_task_obj.fail_task()
                self.jobs[fail_task_obj.jobId].fail_one_task(fail_taskid)

    def add_executor(self, plan):
        logger.debug('create executors,plan is :{}'.format(plan))
        total_n = len(plan)
        failed_n = 0
        failed_executorids = []
        for plan_item in plan:
            executor_id = plan_item.get('executorId')
            device_id = plan_item.get('deviceId')
            category_id = plan_item.get('categoryId')
            request_cpu = self.taskcategorys[category_id].requestCPU
            request_memory = self.taskcategorys[category_id].requestMemory
            prepare_duration = self.taskcategorys[category_id].prepareDuration
            executor_obj = Executor(executorId=executor_id, createTime=self.walltime, \
                                    categoryId=category_id, requestCPU=request_cpu, requestMemory=request_memory,
                                    prepareDuration=prepare_duration)
            if plan_item['deviceType'] == DeviceType.Cloud.name:
                r = self.hosts[device_id].create_exector(executor_obj)
            elif plan_item['deviceType'] == DeviceType.BS.name:
                r = self.bss[device_id].create_exector(executor_obj)
            else:
                r = self.ues[device_id].create_exector(executor_obj)
            if r:
                result = 'successful'
            else:
                result = 'failed'
                failed_n += 1
                failed_executorids.append(executor_id)
            logger.debug('create executor {},executorid:{},categoryid:{},devicetype:{},deviceid:{}'. \
                         format(result, executor_id, category_id, plan_item['deviceType'], device_id))
        success_n = total_n - failed_n
        logger.debug('create executors result:{}/{} success.failed executorids:{}'.format(success_n, total_n,
                                                                                         failed_executorids))
        return success_n, total_n, failed_executorids

    def delete_executor(self, plan):
        logger.debug('delete executors,plan is :{}'.format(plan))
        for executor_id in plan:
            executor_find = False
            for h in self.hosts:
                if executor_id in h.executorids:
                    executor_obj = list(filter(lambda x: x.executorId == executor_id, h.executors))[0]
                    h.delete_executor(executor_obj)
                    if executor_obj.status == ExecutorStatus.Busy.name:
                        task_obj = self._get_task_obj(executor_obj.taskid)
                        task_obj.fail_task()
                        self.jobs[task_obj.jobId].fail_one_task(task_obj.taskId)
                    executor_find = True
                    break
            if executor_find:
                continue

            for b in self.bss:
                if executor_id in b.executorids:
                    executor_obj = list(filter(lambda x: x.executorId == executor_id, b.executors))[0]
                    b.delete_executor(executor_obj)
                    if executor_obj.status == ExecutorStatus.Busy.name:
                        task_obj = self._get_task_obj(executor_obj.taskid)
                        task_obj.fail_task()
                        self.jobs[task_obj.jobId].fail_one_task(task_obj.taskId)
                    executor_find = True
                    break
            if executor_find:
                continue

            for u in self.ues:
                if executor_id in u.executorids:
                    executor_obj = list(filter(lambda x: x.executorId == executor_id, u.executors))[0]
                    u.delete_executor(executor_obj)
                    if executor_obj.status == ExecutorStatus.Busy.name:
                        task_obj = self._get_task_obj(executor_obj.taskid)
                        task_obj.fail_task()
                        self.jobs[task_obj.jobId].fail_one_task(task_obj.taskId)
                    break

    def task_executor_match(self, plan):
        logger.debug('task_executor match,plan is :{}'.format(plan))
        for plan_item in plan:
            executor_id, task_id = plan_item.get('executorId'), plan_item.get('taskId')
            find_exeutor = self._get_executor_obj(executor_id)
            find_task = self._get_task_obj(task_id)
            if (find_exeutor is None) or (find_task is None):
                continue
            executor_obj, executor_devicetype, executor_deviceid, executor_computefactor \
                = find_exeutor
            task_obj = find_task
            if not task_obj.categoryId == executor_obj.categoryId:
                continue
            if (executor_obj.status != ExecutorStatus.Free.name) or (task_obj.status != TaskStatus.Ready.name) \
                    or (task_obj.arriveTime > self.walltime):
                continue
            executor_obj.exec_task(task_id)
            task_obj.start_task(executor_id, self.walltime, executor_devicetype, executor_deviceid,
                                executor_computefactor)
            self.jobs[task_obj.jobId].start_one_task(self.walltime, task_id)

    def exec_plan(self):
        executor_plan = pd.read_csv(os.path.join(self.plan_path, 'executor.csv'))
        task_plan = pd.read_csv(os.path.join(self.plan_path, 'task.csv'))
        # 所有event发生的时刻
        event_times = sorted(list(set(executor_plan['Time'].values) | set(task_plan['Time'].values)))
        for event_idx, wall_time in enumerate(event_times):
            before_wall_time = self.walltime
            for wt in range(before_wall_time, wall_time, 1):
                self.forward_walltime(wt+1)
            delete_executor_df = executor_plan[
                executor_plan.apply(lambda x: (x['Time'] == wall_time) and (x['Action'] == 'Delete'), axis=1)]
            if (not delete_executor_df.empty):
                delete_executor_plan = delete_executor_df['ExecutorId'].astype(np.int32).values.tolist()
                self.delete_executor(delete_executor_plan)

            add_executor_df = executor_plan[
                executor_plan.apply(lambda x: (x['Time'] == wall_time) and (x['Action'] == 'Add'), axis=1)]
            if not (add_executor_df.empty):
                add_executor_df = add_executor_df[['ExecutorId', 'CategoryId', 'DeviceType', 'DeviceId']]
                add_executor_df[['ExecutorId', 'CategoryId', 'DeviceId']] = add_executor_df[
                    ['ExecutorId', 'CategoryId', 'DeviceId']].astype(np.int32)
                add_executor_df.rename(
                    columns={'ExecutorId': 'executorId', 'CategoryId': 'categoryId', 'DeviceType': 'deviceType',
                             'DeviceId': 'deviceId'}, inplace=True)
                add_executor_plan = eval(add_executor_df.to_json(orient='records'))
                self.add_executor(add_executor_plan)

            executor_match_df = task_plan[task_plan.apply(lambda x: x['Time'] == wall_time, axis=1)]
            if not (executor_match_df.empty):
                executor_match_df = executor_match_df[['ExecutorId', 'TaskId']]
                executor_match_df[['ExecutorId', 'TaskId']] = executor_match_df[['ExecutorId', 'TaskId']].astype(
                    np.int32)
                executor_match_df.rename(columns={'ExecutorId': 'executorId', 'TaskId': 'taskId'}, inplace=True)
                executor_match_plan = eval(executor_match_df.to_json(orient='records'))
                self.task_executor_match(executor_match_plan)

    def get_score(self):
        # 正在running的task执行完成
        last_time = 0
        for task_obj in itertools.chain(
                *[list(filter(lambda x: (x.status == TaskStatus.Running.name) and (
                    all([self._get_task_obj(p[0]).status == TaskStatus.Completed.name for p in
                                            x.parentTasks])),
                              j.tasks)) for j in filter(lambda x: x.status == JobStatus.Running.name, self.jobs)]):
            ptask_params = []
            if len(task_obj.parentTasks) > 0:
                for pt in task_obj.parentTasks:
                    pt_taskid = pt[0]
                    ptask_params.append(self._get_ptask_param(task_obj.jobId, pt_taskid))
            task_expect_end_time = task_obj.get_expect_end_time(
                ptask_params, self.cloud_df, self.host_df, self.bs_df, self.ue_metric_df)
            last_time = max(last_time, task_expect_end_time)
        before_wall_time = self.walltime
        for wt in range(before_wall_time, last_time, 1):
            self.forward_walltime(wt + 1)
        # self.forward_walltime(last_time)
        # 计算分数
        task_complete_rate = round(
            sum([len(list(filter(lambda x: x.status == TaskStatus.Completed.name, j.tasks))) for j in
                 filter(lambda x: x.status != JobStatus.Waiting.name, self.jobs)]) / self.task_num, 2)
        completed_jobs = list(filter(lambda x: x.status == JobStatus.Completed.name, self.jobs))
        job_complete_rate = round(len(completed_jobs) / len(self.jobs), 2)
        avg_jct = round(float(np.mean([j.jct for j in self.jobs])), 2)
        score = task_complete_rate + 20000 * job_complete_rate + (10000 - avg_jct)
        return score


def get_one_score(datasetpath: str, resultpath: str, dn: str) -> float:
    environment = Environment(os.path.join(datasetpath, dn), os.path.join(resultpath, dn))
    environment.exec_plan()
    dn_score = environment.get_score()
    logger.info('data name:{},score:{}'.format(dn, dn_score))
    return dn_score


def get_all_score(datasetpath: str, resultpath: str, datanames: List) -> float:
    score = 0
    max_workers = min(min(cpu_count(), 10), len(datanames))
    logger.info(f"using {max_workers} cpus")
    executor = ThreadPoolExecutor(max_workers=max_workers)
    scores = executor.map(get_one_score, [datasetpath]*len(datanames), [resultpath]*len(datanames), datanames)
    score = round(sum(scores) / len(datanames), 2)
    return score


def parse_input():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('-datasetpath', dest='datasetpath', type=str, required=True,
                        help='Please input test dataset path')
    parser.add_argument('-resultpath', dest='resultpath', type=str, required=True,
                        help='Please input result path')
    inputs = parser.parse_args()
    return inputs


if __name__ == '__main__':
    logger.info('begin to make score')
    inputs = parse_input()
    # 数据集存放路径
    datasetpath = inputs.datasetpath
    # 结果存放路径
    resultpath = inputs.resultpath
    # datanames = ['test{}'.format(i) for i in range(10)]
    datanames = ['test0']
    check_data(resultpath, datanames)
    score = get_all_score(datasetpath, resultpath, datanames)
    logger.info('score is :{}'.format(score))
    logger.info('end to make score')
