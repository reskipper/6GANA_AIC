import os
# 指定新的工作路径
new_path = './'

# 修改工作路径
os.chdir(new_path)

# 打印当前工作路径
print(os.getcwd())

from plan_obj import PlanEnv
from writer import Writer
from log import logger
import os
from application import JobStatus
from tqdm import tqdm

logger.debug('plan begin')

data_type = 'test'

for id in tqdm(range(10)):

    path = '%s%d' % (data_type, id)

    writer = Writer(path)

    data_path = os.path.join('./dataset', path)
    plan_env = PlanEnv(data_path, writer)

    time = 0
    while True:
        plan_env.policy()
        time += 1
        plan_env.forward_walltime(time)
        plan_env.update_task()
        
        if all([job.status == JobStatus.Completed.name for job in plan_env.jobs]):
            break
        
    writer.write()
logger.debug('plan end')