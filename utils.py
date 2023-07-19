

def estimated_runtime(job):
    """
    计算作业的剩余计算量（时间）
    """
    residual_tasks = []
    timer = {}
    for task in job.tasks:
        if task.status in ['Running', 'Completed']:
            timer[task.taskId] = 0
        else:
            residual_tasks.append(task)
            
    if not residual_tasks:
        return 0

    for task in residual_tasks:
        first_task = True
        for p in task.parentTasks:
            if p[0] in [t.taskId for t in residual_tasks]:
                first_task = False
                break
        if first_task:
            timer[task.taskId] = task.computeDuration

    while True:
        for task in residual_tasks:
            if task.taskId in timer.keys():
                continue
            else:  # 还没有算计算量的任务
                ps = [p[0] for p in task.parentTasks]
                if set(ps).issubset(set(timer.keys())):  # 父任务都已经计算了
                    timer[task.taskId] = task.computeDuration + max([timer[p] for p in ps])
        if len(timer) == len(job.tasks):
            break
    
    output = max(timer.values())

    return output

def sum_time(job):
    """
    计算作业的总计算量（时间）
    """
    time = 0
    for task in job.tasks:
        if task.status in ['Waiting', 'Ready']:
            time += task.computeDuration
    
    return time