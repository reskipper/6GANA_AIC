from score import Environment
from application import *
from device import *
from writer import Writer
import itertools
import numpy as np
from log import logger
import math
from utils import *

class PlanEnv(Environment):
    def __init__(self, data_path: str, writer: Writer):
        super().__init__(data_path, None)
        # 事件记录
        self.writer = writer

        # 从self.jobs中提取出所有任务的到达时间
        self.job_arrive_time = set()  # （方便check_event方法中检查是否有新任务到达，并非利用未来信息）
        for job in self.jobs:
            self.job_arrive_time.add(job.arriveTime)

        self.arrived_tasks = {}  # 到达任务
        self.candidate_tasks = {}  # 候选任务  

        self.arrived_jobs = {}  # 到达作业
        self.hostreserved = {}  # 上个时隙云端被要求预留的执行器
        self.bsreserved = {}  # 上个时隙边缘被要求预留的执行器
        self.uereserved = {}  # 上个时隙终端被要求预留的执行器
        # self.host_exeDemand = {}  # 主机的执行器需求

    def check_event(self):
        """
        检查事件
        """
        if self.walltime in self.job_arrive_time:
            # 任务到达
            logger.debug('job arrive')
            self.job_arrive()  # 更新task和job字典

    def job_arrive(self):
        """
        任务到达（更新task和job字典）
        """
        # 从self.jobs中提取出所有到达时间为当前walltime的任务
        arrive_jobs = [job for job in self.jobs if job.arriveTime == self.walltime]
        
        for job in arrive_jobs:
            for task in job.tasks:
                self.arrived_tasks[task.taskId] = task  # 将任务添加到到达任务列表中

            self.arrived_jobs.update({job.jobId: {
                                'est_rt': estimated_runtime(job),
                                'sum_time': sum_time(job),
                                    } for job in arrive_jobs})
            
        self.arrived_jobs = dict(sorted(self.arrived_jobs.items(), key=lambda x: x[1]['est_rt']))  # 排序

    def policy(self):
        """
        在一个时间点进行的决策
        策略
        调用check_event, 获取事件信息，再做处理。
        """
        commands = {
            "add_executor": [],
            "add_task": [],
            "delete_executor": []
        }

        self.check_event()
        self.update_task()

        candidate_tasks_copy = self.candidate_tasks.copy()  # 候选任务 dict
        candidate_tasks_copy = {k:v for k,v in candidate_tasks_copy.items() if v.status == TaskStatus.Ready.name}

        host_jobs_num = 0


        # host ######################################
        host_list = self.hosts
        # 按computerFactor从小到大排序
        host_list = sorted(host_list, key=lambda x: x.computerFactor)

        for host in host_list:
            if candidate_tasks_copy:  # 如果还有候选任务
                candidate_exe_copy = host.executors.copy()  # 候选执行器 list
                free_exe_copy = [exe for exe in candidate_exe_copy if exe.status == ExecutorStatus.Free.name]  # 空闲执行器 list

                exe_used = set()

                exe_reserved = set()

                current_exeid = self.writer.current_executor_id

                host_cpu = host.cpuMargin
                host_memory = host.memoryMargin

                spill = False

                over = False

                for job_id, _ in self.arrived_jobs.items():
                    host_jobs_num += 1
                    to_do_tasks = [task for task in candidate_tasks_copy.values() if task.jobId == job_id]
                    to_do_tasks = sorted(to_do_tasks, key=lambda x: x.computeDuration)  # 按照计算时间从小到大排序
                    to_do_tasks = sorted(to_do_tasks, key=lambda x: len(x.childTasks), reverse=True)  # 按照子任务数量从大到小排序（优先）
                    for task in to_do_tasks:
                        allocated = False

                        # 从self.taskcategorys里找到task.categoryId对应的requestCPU和requestMemory
                        for taskcategory in self.taskcategorys:
                            if taskcategory.categoryId == task.categoryId:
                                task.requestCPU = taskcategory.requestCPU
                                task.requestMemory = taskcategory.requestMemory
                                break

                        for executor in free_exe_copy:
                            # 已存在空闲的所需执行器，直接分配任务，todo<预先规划未来任务>
                            if executor.categoryId == task.categoryId and executor.status == ExecutorStatus.Free.name and executor.executorId not in exe_used:
                                commands['add_task'].append([executor.executorId, task.taskId])
                                exe_used.add(executor.executorId)
                                allocated = True
                                break
                        if not allocated:
                            # 不存在空闲的所需执行器
                            reserve, exeid = self.exe_is_available(candidate_exe_copy, host.computerFactor, task.categoryId, exe_used)
                            if reserve:
                                # 预留执行器
                                exe_used.add(exeid)
                                exe_reserved.add(exeid)
                                allocated = True
                            else:
                                # 尝试新建执行器
                                if host_cpu >= task.requestCPU and host_memory >= task.requestMemory:
                                    commands['add_executor'].append([task.categoryId, 1, 0])
                                    host_cpu -= task.requestCPU
                                    host_memory -= task.requestMemory
                                    exe_used.add(current_exeid)
                                    exe_reserved.add(current_exeid)
                                    current_exeid += 1
                                    allocated = True
                                elif not spill:
                                    # 尝试删除后再新建
                                    can_be_deleted =[exe for exe in free_exe_copy if exe.executorId not in self.hostreserved[host.hostId] and exe.executorId not in exe_used]
                                    can_be_deleted = sorted(can_be_deleted, key=lambda x: x.createTime)  # 按创建时间从小到大排列
                                    while can_be_deleted:  # 有可以删除的执行器
                                        # 删除最早创建的执行器
                                        commands['delete_executor'].append(can_be_deleted[0].executorId)
                                        can_be_deleted.pop(0)
                                        if host_cpu >= task.requestCPU and host_memory >= task.requestMemory:
                                            commands['add_executor'].append([task.categoryId, 1, 0])
                                            host_cpu -= task.requestCPU
                                            host_memory -= task.requestMemory
                                            exe_used.add(current_exeid)
                                            exe_reserved.add(current_exeid)
                                            current_exeid += 1
                                            allocated = True
                                            break
                                    if can_be_deleted == []:  # 没有可以删除的执行器
                                        spill = True
                                
                        if allocated:
                            # 分配成功
                            del candidate_tasks_copy[task.taskId]  # 从candidate_tasks_copy中删除已经分配的任务

                        # 如果free_exe_copy里的执行器的序号都在exe_used里，说明没有空闲的执行器了，直接退出
                        if not [exe for exe in free_exe_copy if exe.executorId not in exe_used]:
                            over = True
                            break
                    
                    if over:
                        break

                self.hostreserved[host.hostId] = exe_reserved  # 记录预留的执行器
        #############################################


        # bs ########################################
        bs_list = self.bss
        # 按computerFactor从小到大排序
        bs_list = sorted(bs_list, key=lambda x: x.computerFactor)

        n = len(host.executors) + len(commands['add_executor']) - len(commands['delete_executor'])  # 当前云端执行器数量

        # bs_jobs = 0

        for bs in bs_list:
            if candidate_tasks_copy:  # 如果还有候选任务
                candidate_exe_copy = bs.executors.copy()  # 候选执行器 list
                free_exe_copy = [exe for exe in candidate_exe_copy if exe.status == ExecutorStatus.Free.name]  # 空闲执行器 list

                exe_used = set()

                exe_reserved = set()

                bs_cpu = bs.cpuMargin
                bs_memory = bs.memoryMargin

                spill = False

                over = False

                bef_job_count = 0  # 总的job数量

                bs_jobs_num = 0  # bs上的job数量

                for job_id, _ in self.arrived_jobs.items():
                    bef_job_count += 1
                    if bef_job_count <= host_jobs_num:
                        continue
                    bs_jobs_num += 1
                    to_do_tasks = [task for task in candidate_tasks_copy.values() if task.jobId == job_id]
                    to_do_tasks = sorted(to_do_tasks, key=lambda x: x.computeDuration)  # 按照计算时间从小到大排序
                    to_do_tasks = sorted(to_do_tasks, key=lambda x: len(x.childTasks), reverse=True)  # 按照子任务数量从大到小排序（优先）

                    for task in to_do_tasks:
                        bs_exe_time = task.computeDuration * bs.computerFactor
                        func = (n * bs_exe_time) / (host_jobs_num * host.computerFactor)  # '水'

                        pool = []
                        pool_ = []

                        for num, j in enumerate(self.arrived_jobs.items()):
                            if len(pool) == host_jobs_num:
                                pool_.append(j[1]['sum_time'])
                            else: 
                                pool.append(j[1]['sum_time'])

                        res_count = 0

                        continue_flag = False

                        while min(pool) < func:
                            minus = min(pool)
                            func = func - minus
                            for b in range(len(pool)):
                                pool[b] = pool[b] - minus
                            pool = [p for p in pool if p != 0]
                            while len(pool) < host_jobs_num:  # 补充job
                                if pool_ == []:
                                    continue_flag = True
                                    break
                                pool.append(pool_.pop(0))
                                res_count += 1
                            if continue_flag:
                                break

                        if continue_flag:
                            continue

                        if res_count >= bs_jobs_num:  # 不可以执行任务 raise
                            continue

                        else:  # 可以执行任务
                            allocated = False

                            # 从self.taskcategorys里找到task.categoryId对应的requestCPU和requestMemory
                            for taskcategory in self.taskcategorys:
                                if taskcategory.categoryId == task.categoryId:
                                    task.requestCPU = taskcategory.requestCPU
                                    task.requestMemory = taskcategory.requestMemory
                                    break

                            # 已存在空闲的所需执行器，直接分配任务，todo<预先规划未来任务>
                            for executor in free_exe_copy:
                                if executor.categoryId == task.categoryId and executor.status == ExecutorStatus.Free.name and executor.executorId not in exe_used:
                                    commands['add_task'].append([executor.executorId, task.taskId])
                                    exe_used.add(executor.executorId)
                                    allocated = True
                                    break
                            if not allocated:
                                # 不存在空闲的所需执行器
                                reserve, exeid = self.exe_is_available(candidate_exe_copy, bs.computerFactor, task.categoryId, exe_used)
                                if reserve:
                                    # 预留执行器
                                    exe_used.add(exeid)
                                    exe_reserved.add(exeid)
                                    allocated = True
                                else:
                                    # 尝试新建执行器
                                    if bs_cpu >= task.requestCPU and bs_memory >= task.requestMemory:
                                        commands['add_executor'].append([task.categoryId, 2, bs.bsId])
                                        bs_cpu -= task.requestCPU
                                        bs_memory -= task.requestMemory
                                        exe_used.add(current_exeid)
                                        exe_reserved.add(current_exeid)
                                        current_exeid += 1
                                        allocated = True
                                    elif not spill:
                                        # 尝试删除后再新建
                                        can_be_deleted =[exe for exe in free_exe_copy if exe.executorId not in self.bsreserved[bs.bsId] and exe.executorId not in exe_used]
                                        can_be_deleted = sorted(can_be_deleted, key=lambda x: x.createTime)  # 按创建时间从小到大排列
                                        while can_be_deleted:  # 有可以删除的执行器
                                            # 删除最早创建的执行器
                                            commands['delete_executor'].append(can_be_deleted[0].executorId)
                                            can_be_deleted.pop(0)
                                            if bs_cpu >= task.requestCPU and bs_memory >= task.requestMemory:
                                                commands['add_executor'].append([task.categoryId, 2, bs.bsId])
                                                bs_cpu -= task.requestCPU
                                                bs_memory -= task.requestMemory
                                                exe_used.add(current_exeid)
                                                exe_reserved.add(current_exeid)
                                                current_exeid += 1
                                                allocated = True
                                                break
                                        if can_be_deleted == []:
                                            spill = True
                                            break
                            if allocated:
                                # 分配成功
                                del candidate_tasks_copy[task.taskId]  # 从candidate_tasks_copy中删除已经分配的任务
                        
                            # 如果free_exe_copy里的执行器的序号都在exe_used里，说明没有空闲的执行器了，直接退出
                            if not [exe for exe in free_exe_copy if exe.executorId not in exe_used]:
                                over = True
                                break
                
                    if over:
                        break

                # bs_jobs = bs_jobs_num

                self.bsreserved[bs.bsId] = exe_reserved
        #############################################


        # ue ########################################
        ue_list = self.online_ues
        ue_list = sorted(ue_list, key=lambda x: x.rate, reverse=True)
        ue_list = sorted(ue_list, key=lambda x: x.computerFactor)

        all_over = True

        for ue in ue_list:
            if all_over == True and ue != ue_list[0]:
                break

            if candidate_tasks_copy:  # 如果还有候选任务
                candidate_exe_copy = ue.executors.copy()
                free_exe_copy = [exe for exe in candidate_exe_copy if exe.status == ExecutorStatus.Free.name]  # 空闲执行器 list

                exe_used = set()  # 已经使用的执行器

                exe_reserved = set()  # 预留的执行器

                ue_cpu = ue.cpuMargin
                ue_memory = ue.memoryMargin

                spill = False  # 是否溢出

                over = False  # 是否结束

                bef_job_count = 0

                ue_jobs_num = 0

                for job_id, _ in self.arrived_jobs.items():
                    bef_job_count += 1
                    if bef_job_count <= host_jobs_num:
                        continue
                    ue_jobs_num += 1
                    to_do_tasks = [task for task in candidate_tasks_copy.values() if task.jobId == job_id]
                    to_do_tasks = sorted(to_do_tasks, key=lambda x: x.outputsize)
                    to_do_tasks = sorted(to_do_tasks, key=lambda x: x.computeDuration)

                    continue_flag = False
                    
                    for task in to_do_tasks:
                        if ue.rate == 0:
                            continue_flag = True
                            break
        
                        trans_time = task.outputsize / ue.rate
                        trans_time = math.ceil(trans_time)
                        ue_exe_time = task.computeDuration * ue.computerFactor + trans_time * 2

                        func = (n * ue_exe_time) / (host_jobs_num * host.computerFactor)

                        pool = []
                        pool_ = []

                        for num, j in enumerate(self.arrived_jobs.items()):
                            if len(pool) == host_jobs_num:
                                pool_.append(j[1]['sum_time'])
                            else: 
                                pool.append(j[1]['sum_time'])

                        res_count = 0

                        continue_flag = False

                        while min(pool) < func:
                            minus = min(pool)
                            func = func - minus
                            for b in range(len(pool)):
                                pool[b] = pool[b] - minus
                            pool = [p for p in pool if p != 0]
                            while len(pool) < host_jobs_num:  # 补充job
                                if pool_ == []:
                                    continue_flag = True
                                    break
                                pool.append(pool_.pop(0))
                                res_count += 1
                            if continue_flag:
                                break

                        if continue_flag:
                            continue

                        if res_count >= ue_jobs_num:  # 不可以执行任务 raise
                            continue

                        else:  # 可以执行任务
                            all_over = False
                            allocated = False

                            # 从self.taskcategorys里找到task.categoryId对应的requestCPU和requestMemory
                            for taskcategory in self.taskcategorys:
                                if taskcategory.categoryId == task.categoryId:
                                    task.requestCPU = taskcategory.requestCPU
                                    task.requestMemory = taskcategory.requestMemory
                                    break

                            # 已存在空闲的所需执行器，直接分配任务，todo<预先规划未来任务>
                            for executor in free_exe_copy:
                                if executor.categoryId == task.categoryId and executor.status == ExecutorStatus.Free.name and executor.executorId not in exe_used:
                                    commands['add_task'].append([executor.executorId, task.taskId])
                                    exe_used.add(executor.executorId)
                                    allocated = True
                                    break
                            if not allocated:
                                # 不存在空闲的所需执行器
                                reserve, exeid = self.exe_is_available(candidate_exe_copy, ue.computerFactor, task.categoryId, exe_used)
                                if reserve:
                                    # 预留执行器
                                    exe_used.add(exeid)
                                    exe_reserved.add(exeid)
                                    allocated = True
                                else:
                                    # 尝试新建执行器
                                    if ue_cpu >= task.requestCPU and ue_memory >= task.requestMemory:
                                        commands['add_executor'].append([task.categoryId, 3, ue.ueId])
                                        ue_cpu -= task.requestCPU
                                        ue_memory -= task.requestMemory
                                        exe_used.add(current_exeid)
                                        exe_reserved.add(current_exeid)
                                        current_exeid += 1
                                        allocated = True
                                    elif not spill:
                                        # 尝试删除后再新建
                                        can_be_deleted =[exe for exe in free_exe_copy if exe.executorId not in self.uereserved[ue.ueId] and exe.executorId not in exe_used]
                                        can_be_deleted = sorted(can_be_deleted, key=lambda x: x.createTime)  # 按创建时间从小到大排列
                                        while can_be_deleted:  # 有可以删除的执行器
                                            # 删除最早创建的执行器
                                            commands['delete_executor'].append(can_be_deleted[0].executorId)
                                            can_be_deleted.pop(0)
                                            if ue_cpu >= task.requestCPU and ue_memory >= task.requestMemory:
                                                commands['add_executor'].append([task.categoryId, 3, ue.ueId])
                                                ue_cpu -= task.requestCPU
                                                ue_memory -= task.requestMemory
                                                exe_used.add(current_exeid)
                                                exe_reserved.add(current_exeid)
                                                current_exeid += 1
                                                allocated = True
                                                break
                                        if can_be_deleted == []:
                                            spill = True
                                            break
                            if allocated:
                                # 分配成功
                                del candidate_tasks_copy[task.taskId]  # 从candidate_tasks_copy中删除已经分配的任务
                        
                            # 如果free_exe_copy里的执行器的序号都在exe_used里，说明没有空闲的执行器了，直接退出
                            if not [exe for exe in free_exe_copy if exe.executorId not in exe_used]:
                                over = True
                                break
                    
                    if over or continue_flag:
                        break

                self.uereserved[ue.ueId] = exe_reserved                           
        #############################################


        self.action(commands)

    def update_task(self):
        """
        更新task和job信息
        """
        # 删除候选任务中已经完成的任务
        to_be_deleted = []
        for task in self.candidate_tasks.values():
            if task.status == TaskStatus.Completed.name:
                to_be_deleted = to_be_deleted + [task.taskId]
                est = estimated_runtime(self.jobs[task.jobId])
                if est == 0 and self.jobs[task.jobId].status == JobStatus.Completed.name:
                    if task.jobId in self.arrived_jobs.keys():
                        del self.arrived_jobs[task.jobId]
                        print('time: %d, job_num:' % self.walltime, len(self.arrived_jobs))
                        print(self.arrived_jobs)
                else:
                    self.arrived_jobs[task.jobId]['est_rt'] = est
                    job_time = sum_time(self.jobs[task.jobId])
                    self.arrived_jobs[task.jobId]['sum_time'] = job_time
                    self.arrived_jobs = dict(sorted(self.arrived_jobs.items(), key=lambda x: x[1]['est_rt']))

        for task_id in to_be_deleted:
            del self.candidate_tasks[task_id]

        # 将到达任务中的就绪任务添加到候选任务中
        to_be_deleted = []
        for task in self.arrived_tasks.values():
            if task.status == TaskStatus.Ready.name:
                candidate = True
                for parent_task in task.parentTasks:
                    if (parent_task[0] in self.candidate_tasks.keys()) or (parent_task[0] in self.arrived_tasks.keys()):
                        candidate = False
                        break
                if candidate:
                    self.candidate_tasks[task.taskId] = task
                    to_be_deleted = to_be_deleted + [task.taskId]
        for task_id in to_be_deleted:
            del self.arrived_tasks[task_id]

    def action(self, commands: dict):
        """
        动作
        分别是添加执行器；删除执行器；添加任务
        """
        # 删除执行器
        if commands['delete_executor']:
            self.delete_executor_p(commands['delete_executor'])
        # 添加执行器
        if commands['add_executor']:
            self.add_executor_p(commands['add_executor'])
        # 添加任务
        if commands['add_task']:
            self.add_task_p(commands['add_task'])
            
    def add_executor_p(self, commands: list):
        """
        添加执行器，包括记录和在环境中操作
        需要添加的执行器信息包括：
        CategoryId, DeviceType, DeviceId
        通过commands参数传入，格式为[(CategoryId, DeviceType, DeviceId), ...]
        writer.add_executor(1, 1, 1, 1)  即 time=1，category_id=1，device_type=Cloud，device_id=1
        """
        for category_id, device_type, device_id in commands:
            self.writer.add_executor_w(self.walltime, category_id, device_type, device_id)
        executor_plan = self.writer.executor_df
        
        # 找出所有在当前 wall time 需要被添加的执行器，并将其添加到系统中
        add_executor_df = executor_plan[
            executor_plan.apply(lambda x: (x['Time'] == self.walltime) and (x['Action'] == 'Add'), axis=1)]
        if not (add_executor_df.empty):
            add_executor_df = add_executor_df[['ExecutorId', 'CategoryId', 'DeviceType', 'DeviceId']]
            add_executor_df[['ExecutorId', 'CategoryId', 'DeviceId']] = add_executor_df[
                ['ExecutorId', 'CategoryId', 'DeviceId']].astype(np.int32)
            add_executor_df.rename(
                columns={'ExecutorId': 'executorId', 'CategoryId': 'categoryId', 'DeviceType': 'deviceType',
                         'DeviceId': 'deviceId'}, inplace=True)
            add_executor_plan = eval(add_executor_df.to_json(orient='records'))
            self.add_executor(add_executor_plan)

    def delete_executor_p(self, commands: list):
        """
        删除执行器，包括记录和在环境中操作
        需要删除的执行器信息包括：
        ExecutorId
        通过commands参数传入，格式为[ExecutorId, ...]
        writer.delete_executor(1, 1)  即 time=1，executor_id=1
        """
        for executor_id in commands:
            self.writer.delete_executor_w(self.walltime, executor_id)
        executor_plan = self.writer.executor_df

        # 找出所有在当前 wall time 需要被删除的执行器，并将其从系统中移除
        delete_executor_df = executor_plan[
            executor_plan.apply(lambda x: (x['Time'] == self.walltime) and (x['Action'] == 'Delete'), axis=1)]
        if (not delete_executor_df.empty):
            delete_executor_plan = delete_executor_df['ExecutorId'].astype(np.int32).values.tolist()
            self.delete_executor(delete_executor_plan)

    def add_task_p(self, commands: list):
        """
        添加任务，包括记录和在环境中操作
        需要添加的任务信息包括：
        ExecutorId, TaskId
        通过commands参数传入，格式为[(ExecutorId, TaskId), ...]
        writer.add_task(1, 1, 1)  即 time=1，executor_id=1，task_id=1
        """
        for executor_id, task_id in commands:
            self.writer.add_task_w(self.walltime, executor_id, task_id)
        task_plan = self.writer.task_df

        # 在当前 wall time，找出需要被分配的任务，并将任务分配给相应的执行器
        executor_match_df = task_plan[task_plan.apply(lambda x: x['Time'] == self.walltime, axis=1)]
        if not (executor_match_df.empty):
            executor_match_df = executor_match_df[['ExecutorId', 'TaskId']]
            executor_match_df[['ExecutorId', 'TaskId']] = executor_match_df[['ExecutorId', 'TaskId']].astype(
                np.int32)
            executor_match_df.rename(columns={'ExecutorId': 'executorId', 'TaskId': 'taskId'}, inplace=True)
            executor_match_plan = eval(executor_match_df.to_json(orient='records'))
            self.task_executor_match(executor_match_plan)

    def exe_is_available(self, executors, cf, cat, used):
        """
        获取未来时刻执行器可用性（已有的非free执行器）

        return: 可用性 0-1, executorid
        """
        available_executors = executors.copy()
        available_executors = [executor for executor in available_executors 
                               if executor.categoryId == cat 
                               and executor.executorId not in used]  # 当前时刻非free的执行器
        selected_executors = {}  # executorid: 剩余时间
        for executor in available_executors:
            if executor.status == ExecutorStatus.Busy.name:
                r_time = [(task.startTime + (task.computeDuration*cf) - self.walltime) for taskid, task in self.candidate_tasks.items() if taskid == executor.taskid][0]
            elif executor.status == ExecutorStatus.Preparing.name:
                r_time = executor.createTime + executor.prepareDuration - self.walltime
            selected_executors[executor.executorId] = r_time
        if selected_executors:
            selected_executors = sorted(selected_executors.items(), key=lambda x: x[1], reverse=False)
            perpDur = [c.prepareDuration for c in self.taskcategorys if c.categoryId == cat][0]
            if selected_executors[0][1] <= perpDur + 5:
                return True, selected_executors[0][0]
            else:
                return False, None
        else:
            return False, None

    @property
    def online_ues(self):
        """
        在线用户
        """
        current_time = self.walltime % 1200
        online_ues = [ue for ue in self.ues if (ue.onlineTime <= current_time) and (ue.offlineTime >= current_time)]
        return online_ues
    
    @property
    def arrived_tasks_num(self):
        """
        剩余到达任务数量
        """
        return len(self.arrived_tasks)
    
    @property
    def candidate_tasks_num(self):
        """
        剩余候选任务数量
        """
        return len(self.candidate_tasks)

    @property
    def executors(self):
        """
        执行器信息
        """
        executors_list = []
        for host in self.hosts:
            executors_list = executors_list + host.executors
        for bs in self.bss:
            executors_list = executors_list + bs.executors
        for ue in self.ues:
            executors_list = executors_list + ue.executors
        return executors_list
