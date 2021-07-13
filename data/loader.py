import pandas as pd
import os

from framework.instance import InstanceConfig
from framework.machine import MachineConfig


# def MachineConfigLoader(filename):  # id, cpu_capacity, memory_capacity, disk_capacity...
#     machine_configs = []
#     with open(filename) as f:
#         csv_reader = csv.reader(f)
#         for row in csv_reader:
#             machine_id, cpu_capacity, memory_capacity, disk_capacity = row[0], float(row[1]), float(row[2]), float(row[3])
#             machine_configs.append(MachineConfig(machine_id, cpu_capacity, memory_capacity, disk_capacity))
#     return machine_configs


# def InstanceConfigLoader(filename):
#     instance_configs = []
#     with open(filename) as f:
#         csv_reader = csv.reader(f)
#         for row in csv_reader:
#             inst_id, machine_id, cpu_curve, memory_curve, disk = row[0], row[2], [float(cpu) for cpu in row[3].split('|')], [float(memory) for memory in row[4].split('|')], float(row[5])
#             instance_configs.append(InstanceConfig(inst_id, machine_id, cpu_curve[0], memory_curve[0], disk, cpu_curve, memory_curve))
#     return instance_configs


def InstanceConfigLoader(vm_cpu_request_file, vm_machine_id_file, vm_cpu_utils_folder):
    instance_configs = []
    vm_cpu_requests = pd.read_csv(vm_cpu_request_file, header = None).values.squeeze()
    vm_machine_ids = pd.read_csv(vm_machine_id_file, header = None)[1].values
    vm_cpu_utils = sorted([os.path.join(vm_cpu_utils_folder, x) for x in os.listdir(vm_cpu_utils_folder)])
    
    assert len(vm_cpu_requests) == len(vm_machine_ids)
    
    for i in range(len(vm_cpu_requests)):
        cpu_curve = pd.read_csv(vm_cpu_utils[i], header = None).values.squeeze() * vm_cpu_requests[i]
        memory_curve = np.zeros_like(cpu_curve)
        disk_curve = np.zeros_like(cpu_curve)
        instance = InstanceConfig(i, vm_machine_ids[i], cpu_curve[0], 0, disk, cpu_curve, memory_curve)
        instance_configs.append(instance)
        
    return instance_configs