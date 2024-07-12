#!/usr/bin/python3

import os
import subprocess
import re
from typing import List, Optional
from datetime import datetime
from typing import Callable

# Naming conventions :
# - Run : 1 execution of the measured program
# - batch : N runs 
# - Benchmark : N batch, each in a different configuration

RESULTS_DIR_PATH = "/root/results"
NFS_RESULTS_DIR_PATH = "/home/cgachod/results"
TMP_SH_FILE_PATH = "/tmp/.experiments.sh"

NUMA_BALANCING_CTL = "/proc/sys/kernel/numa_balancing"
SPLIT_NODE_SHARED_HUGE_PAGES_CTL = "/proc/sys/kernel/nb_split_shared_hugepages"

os.makedirs(RESULTS_DIR_PATH, exist_ok=True)
os.makedirs(NFS_RESULTS_DIR_PATH, exist_ok=True)


def get_shell_command_output(command: str):
    return subprocess.run(
        command,
        shell=True, stdout = subprocess.PIPE, universal_newlines = True
    ).stdout.strip('\n')

def get_machine_name() -> str:
    m = re.match(r"^[^\.]+", get_shell_command_output("hostname"))
    if not m :
        raise Exception("Unable to get machine name")
    return m.group()

MACHINE_NAME = get_machine_name()




class CommandChain:
    def __init__(self, lines: List[str] = None) -> None:
        self.lines = lines if lines is not None else []
        
    def append(self, new_line: str):
        self.lines.append(new_line)
        
    def add(self, other: "CommandChain"):
        self.lines += other.lines
        
    def __add__(self, o):
        return CommandChain(self.lines + o.lines)
        
    def chain(self):
        return " && ".join(self.lines)
        
    def execute(self, debug = False, debug_lines = 100):
        with open(TMP_SH_FILE_PATH, 'w') as f:
            f.write("#!/bin/bash\n\n")
            
            for l in self.lines:
                f.write(l)
                f.write("\n\n")
                
        if debug:
            print(f"{debug_lines} first lines written to {TMP_SH_FILE_PATH}")
            with open(TMP_SH_FILE_PATH, 'r') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    stripped_line = line.strip('\n')
                    if len(stripped_line) == 0 :
                        continue
                    print(stripped_line)
                    if line_count >= debug_lines :
                        return
        
        
        subprocess.run(f"chmod +x {TMP_SH_FILE_PATH}", shell=True)
        subprocess.run(f"{TMP_SH_FILE_PATH}", shell=True)



def get_perf_stat_cmd_old(program: str, events: str, output_path: str) -> str:
    return f"perf stat -A -a -j -e {events} -o {output_path} --append {program} >> {output_path}"

def get_perf_stat_cmd(program: str, events: str, output_path: str) -> str:
    time_events = "duration_time,user_time,system_time"
    return f"perf stat -A -a -j -e {events},{time_events} -D 100 -o {output_path} --append {program} >> {output_path}"


def get_batch(program: str, measurement_command: str, nwarmups: int = 0) -> CommandChain:
    chain = CommandChain([program] * nwarmups)
    chain.append(measurement_command)
    return chain






def configure_numa_balancing(on = False):
    return CommandChain([f"echo {int(on)} > {NUMA_BALANCING_CTL}"])


def get_sockorder_omp_places():
    matches = re.findall(
        r"NUMA +node\d+ +CPU\(s\): +([\d,]+)", 
        subprocess.run("lscpu", stdout = subprocess.PIPE, universal_newlines = True).stdout
    )
    
    node_places = [",".join([f"{{{cpuid}}}" for cpuid in cpulist.split(",")]) for cpulist in matches]
    # result = ""
    # for cpulist in matches:
    #     result += ",".join([f"{{{cpuid}}}" for cpuid in cpulist.split(",")])
    return ",".join(node_places)

def get_sequential_omp_places():
    # nproc = subprocess.run("nproc", stdout = subprocess.PIPE, universal_newlines = True).stdout
    return f"{{0}}:{get_shell_command_output('nproc')}:1"

def configure_omp_pinning(omp_places: Optional[str]) -> CommandChain:
    if omp_places is None :
        return CommandChain([
            "export OMP_PROC_BIND=false", 
            "unset OMP_PLACES"])
        
    return CommandChain([
        "export OMP_PROC_BIND=true",
        f"export OMP_PLACES={omp_places}"
    ])
    
    
    
THP_CONFIG_VALUES = ["always", "never", "madvise"]
def configure_thp(enabled: str, defrag: str):
    if enabled not in THP_CONFIG_VALUES:
        raise Exception(f"enabled param should be in {THP_CONFIG_VALUES}")
    if defrag not in THP_CONFIG_VALUES:
        raise Exception(f"defrag param should be in {THP_CONFIG_VALUES}")
    return CommandChain([
        f"echo {enabled} > /sys/kernel/mm/transparent_hugepage/enabled",
        f"echo {defrag} > /sys/kernel/mm/transparent_hugepage/defrag"
    ])
    


def echo_config():
    return CommandChain([
        f"echo Numa Balancing enabled : $(cat {NUMA_BALANCING_CTL})",
        "echo OMP_PROC_BIND : $OMP_PROC_BIND",
        "echo OMP_PLACES : $OMP_PLACES",
        "echo THP enabled : $(cat /sys/kernel/mm/transparent_hugepage/enabled)",
        "echo THP defrag : $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
    ])

    

def get_kernel_version():
    return get_shell_command_output('uname -r')

def get_performance_governor():
    return get_shell_command_output('cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq')

def get_date():
    return datetime.now().strftime('%Y-%m-%d')


def get_config_info(program_name: str):
    return f"{program_name}__{MACHINE_NAME}__v{get_kernel_version()}__{get_performance_governor()}__{get_date()}"


def persist_machine_info(parent_directory: Optional[str] = None):
    lines = []
    if parent_directory is not None:
        lines = [
            f"mkdir -p {parent_directory}/meta",
            f"cd {parent_directory}"
        ]
    else :
        parent_directory = "."
        
    command_chain = CommandChain(lines + [
        f"lscpu > {parent_directory}/meta/lscpu.log",
        f"getconf PAGESIZE > {parent_directory}/meta/pagesize.log",
        f"cat /proc/zoneinfo > {parent_directory}/meta/zoneinfo.log"
    ])
    command_chain.execute()



def get_next_file_id(directory, prefix, extension):
    highest_iteration = 0
    pattern = re.compile(rf"{re.escape(prefix)}__([0-9]+){re.escape(extension)}")

    for file_name in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, file_name)):
            continue
        match = pattern.search(file_name)
        if match:
            iteration = int(match.group(1))
            if iteration > highest_iteration:
                highest_iteration = iteration

    return highest_iteration + 1



# TODO Keep the date stable accross days of running
# Would be nice here to have the name of the folder ?
# Callable that takes as an argument only an id number
def generate_batch(program_path: str, data_dir_name: str, action_name: str, extension: str, measurement_function: Callable, nruns: int, nwarmups: int):
    program_name = os.path.basename(program_path)
    config_info = get_config_info(program_name)
    save_dir_path = RESULTS_DIR_PATH + "/" + config_info + "/" + data_dir_name
    os.makedirs(save_dir_path, exist_ok=True)
    
    prefix = config_info + "__" + data_dir_name + "__" + action_name
    initial_id = get_next_file_id(save_dir_path, prefix, extension)
    
    # Saving the measurement command in the save directory to be able to look at it later
    sample_measurement_cmd = measurement_function(initial_id, save_dir_path, prefix, extension)
    with open(os.path.join(save_dir_path, "sample_measure_command")) as f:
        f.write(sample_measurement_cmd)
        f.write("\n")
    
    run_cmds = []
    for i in range(nruns):
        run_cmds.append(rf"printf '\nStarting run {i}/{nruns}\n'")
        # For more abstraction, following could be turned into a partial / lambda with only the i as parameter
        run_cmds.append("set -x")
        run_cmds.append(measurement_function(initial_id + i, save_dir_path, prefix, extension))
        run_cmds.append("set +x")
    
    return CommandChain([program_path] * nwarmups + run_cmds)

def generate_perf_stat_batch(program_path: str, nruns: int, nwarmups: int, save_data_dir: str, events):
    measurement_func = lambda idx, save_dir_path, prefix, extension : get_perf_stat_cmd(program_path, events, f"{save_dir_path}/{prefix}__{idx}{extension}")
    return generate_batch(program_path, save_data_dir, "perf_stat", ".txt", measurement_func, nruns, nwarmups)








#
#
# Actual experiment functions
#
#


# Measures the number of l3 misses
def EXPERIMENT_benchmark_perf_l3_miss(program_path: str, nruns: int, nwarmups: int, none_coeff = 1.0):
    events = (
        "mem_load_l3_miss_retired.local_dram,"
        "mem_load_l3_miss_retired.remote_dram,"
        "mem_load_l3_miss_retired.remote_fwd,"
        "mem_load_l3_miss_retired.remote_hitm"
    )
    
    def run_batch(config_name, nb_on, omp_places, batch_nruns):
        command_chain = configure_numa_balancing(on = nb_on)
        command_chain += configure_omp_pinning(omp_places)
        command_chain += echo_config()
        command_chain += generate_perf_stat_batch(program_path, batch_nruns, nwarmups, config_name, events)
        print("\n")
        command_chain.execute(debug = False)
        get_shell_command_output(f"cp -r {RESULTS_DIR_PATH}/* {NFS_RESULTS_DIR_PATH}")
        print("\n\n")
        
    run_batch("nb-disabled-sockorder", False, get_sockorder_omp_places(), nruns)
    # run_batch("nb-enabled-sockorder", True, get_sockorder_omp_places(), nruns)
    
    run_batch("nb-disabled-sequential", False, get_sequential_omp_places(), nruns)
    # run_batch("nb-enabled-sequential", True, get_sequential_omp_places(), nruns)
    
    run_batch("nb-disabled-none", False, None, nruns * none_coeff)
    # run_batch("nb-enabled-none", True, None, nruns * none_coeff)
    



def init_command_chain_with_config(nb_on: bool, omp_places: Optional[str]):
    command_chain = configure_numa_balancing(on = nb_on)
    command_chain += configure_omp_pinning(omp_places)
    command_chain += echo_config()
    return command_chain

def init_command_chain_with_config_thp(nb_on: bool, omp_places: Optional[str], thp_enabled: str, thp_defrag: str):
    command_chain = configure_thp(thp_enabled, thp_defrag)
    command_chain += configure_numa_balancing(nb_on)
    command_chain += configure_omp_pinning(omp_places)
    command_chain += echo_config()
    return command_chain


# perf record -v -a -d -W -e "kmem:mm_page_alloc" -e cpu/mem-loads,period=1000/P --phys-data ~/npb/NPB3.4-OMP/bin/cg.B.x
# perf record -v -a -d -W -e "kmem:\*" -e cpu/mem-loads,period=1000/P --phys-data ~/npb/NPB3.4-OMP/bin/cg.B.x
# perf record  -e "kmem:\*" -e cpu/mem-loads,period=1000/P -F 10000 --phys-data ~/npb/NPB3.4-OMP/bin/cg.B.x
# perf script 
# perf script -L -c cg.B.x -F +period
def experiment_memory_allocation_and_acces_in_sequential_pinning(program_path: str, warmups: int = 1):
    home_directory = "/root/tests"
    mem_events_period = 1000
    mem_events = f"cpu/mem-loads,period={mem_events_period}/P,cpu/mem-stores,period={mem_events_period}/P"
    # alloc_events = "kmem:mm_page_alloc"   
    alloc_events = "kmem:\*,syscalls:sys_enter_mmap"
    
    persist_machine_info(home_directory)
    
    def run_benchmark_for_setup(nb_on: bool, omp_places: str, data_file_name: str):
        measure_command = f"perf record -v -a -d -W -e {alloc_events} -e {mem_events} --output {data_file_name} --phys-data {program_path}"
        command_chain = CommandChain([f"cd {home_directory}"])
        command_chain += init_command_chain_with_config(nb_on, omp_places)
        command_chain += CommandChain([program_path] * warmups)
        command_chain.append(measure_command)
        command_chain.execute()
        
    run_benchmark_for_setup(False, get_sequential_omp_places(), "perf-alloc-sequential.data")
    run_benchmark_for_setup(False, get_sockorder_omp_places(), "perf-alloc-sockorder.data")




def EXPERIMENT_mem_access_alloc_l3_miss(program_path: str, warmups: int = 1, capture_l3_miss_local_dram = True):
    home_directory = "/root/tests"
    mem_events_period = 2000
    l3_miss_events_period = 200
    
    mem_events = f"cpu/mem-loads,period={mem_events_period}/P,cpu/mem-stores,period={mem_events_period}/P"
    
    # Following events will always be recorded with a period of 1
    alloc_events = "kmem:\*,syscalls:sys_enter_mmap"
    
    l3_miss_events = (
        "mem_load_l3_miss_retired.remote_dram:P,"
        "mem_load_l3_miss_retired.remote_fwd:P,"
        "mem_load_l3_miss_retired.remote_hitm:P"
    )
    
    if capture_l3_miss_local_dram:
        l3_miss_events += ",mem_load_l3_miss_retired.local_dram:P"
    
    def run_benchmark_for_setup(nb_on: bool, omp_places: Optional[str], data_file_name: str):
        output_file_name = os.path.splitext(data_file_name)[0] + ".output.txt"
        measure_command = f"perf record -v -a -d -W -e {alloc_events} -e {mem_events} -c {l3_miss_events_period} -e {l3_miss_events} --output {data_file_name} --phys-data {program_path} > {output_file_name}"
        command_chain = CommandChain([f"cd {home_directory}"])
        command_chain += init_command_chain_with_config(nb_on, omp_places)
        command_chain += CommandChain([program_path] * warmups)
        command_chain.append(measure_command)
        command_chain.execute()
        
    persist_machine_info(home_directory)
    run_benchmark_for_setup(False, get_sequential_omp_places(), "perf-mem-all-sequential.data")
    run_benchmark_for_setup(False, get_sockorder_omp_places(), "perf-mem-all-sockorder.data")
    
    
# TODO Have something where you specify the variables and it creates a grid automatically ?
# Like 
# - param 1 name, [param 1 values]
# - param 2 name, [param 2 values]
# - 


def arg_dict_to_run_name(arg_dict: dict):
    str_args = [f"{key}-{value}" for key, value in arg_dict.items()]
    return "__".join(str_args)


# parameter values should be : [[("no", False), ("yes", True)], [] ... ]
def grid_experiment(parameters: List[str], parameter_values: List[List[tuple]], run_function: callable):
    n = len(parameter_values)
    if len(parameter_values) != n :
        raise Exception("parameters and parameter_values should always have the same length")
    
    total_runs = 1
    for pv in parameter_values:
        total_runs *= len(pv)
    print(f"Performing grid experiment. Total number of configs : {total_runs}")
    
    curr_params: List[tuple] = []
    
    def arg_dict_from_curr_params():
        return {parameters[i]: curr_params[i][1] for i in range(n)}
    
    def name_from_curr_params_old():
        params_str = [f"{parameters[i]}-{curr_params[i][0]}" for i in range(n)]
        return "__".join(params_str)
    
    def name_from_curr_params():
        params_str = [curr_params[i][0] for i in range(n)]
        return "-".join(params_str)
    
    def recu_explore(idx: int):
        if idx == n:
            arg_dict = arg_dict_from_curr_params()
            arg_dict["name"] = name_from_curr_params()
            run_function(**arg_dict)
            return
        
        for val in parameter_values[idx]:
            curr_params.append(val)
            recu_explore(idx + 1)
            curr_params.pop()
    
    recu_explore(0)        
    
    
def EXPERIMENT_quick_comparison(program_path: str, runs: int = 10, warmups: int = 1):
    home_directory = "~/tests"
    program_name = os.path.splitext(os.path.basename(program_path))[0]
    data_save_path = os.path.join(home_directory, program_name)
    
    def run_with_params(name: str, nb: bool, ompPlaces: Optional[str], thpE: str, thpD: str):
        # print(f"Running with params : name = {name}, nb = {nb}, ompPlaces = {ompPlaces}, thpE = {thpE}, thpD = {thpD}")
        command_chain = init_command_chain_with_config_thp(nb, ompPlaces, thp_enabled=thpE, thp_defrag=thpD)
        measure_command = f"hyperfine --warmup {warmups} --runs {runs} --export-json {os.path.join(data_save_path, name)}.json -n '{name}' '{program_path}'"
        command_chain.append(measure_command)
        command_chain.execute()
    
    # def run_setup(name: str, nb_on: bool, omp_places: Optional[str], thp_enabled: str, thp_defrag: str):
    #     command_chain = init_command_chain_with_config_thp(nb_on, omp_places, thp_enabled, thp_defrag)
    #     measure_command = f"hyperfine --warmup {warmups} --runs {runs} --export-json {os.path.join(home_directory, name)}.json -n '{name}' '{program_path}'"
    #     command_chain.append(measure_command)
    #     command_chain.execute()
        
    persist_machine_info(home_directory)
    grid_experiment(
        ["nb", "ompPlaces", "thpE", "thpD"],
        [
            [("on", True), ("off", False)],
            [("sequential", get_sequential_omp_places()), ("sockorder", get_sockorder_omp_places()), ("none", None)],
            [("never", "never"), ("always", "always")],
            [("never", "never"), ("always", "always")],
        ],
        run_with_params
    )
    
    
    
    
    
    
def EXPERIMENT_benchmark_different_config(program_path: str, runs: int = 20, warmups: int = 2):
    home_directory = "/root/results/new"
    
    program_name = os.path.basename(program_path)
    config_info = get_config_info(program_name)
    save_parent_path = home_directory + "/" + config_info
    print(f"save_parent_path : {save_parent_path}")
    os.makedirs(save_parent_path, exist_ok=True)
    
    # prefix = config_info + "__" + data_dir_name + "__" + action_name
    # initial_id = get_next_file_id(save_dir_path, prefix, extension)
    
    # program_name = os.path.splitext(os.path.basename(program_path))[0]
    # data_save_path = os.path.join(home_directory, program_name)
    
    def run_with_params(name: str, nb: bool, ompPlaces: Optional[str]):
        print(name)
        # print(f"Running with params : name = {name}, nb = {nb}, ompPlaces = {ompPlaces}, thpE = {thpE}, thpD = {thpD}")
        # command_chain = init_command_chain_with_config_thp(nb, ompPlaces, thp_enabled=thpE, thp_defrag=thpD)
        filename = config_info + "__" + name + ".json"
        filedir = os.path.join(save_parent_path, name)
        filepath = os.path.join(filedir, filename)
        os.makedirs(filedir, exist_ok=True)
        print(f"filedir : {filedir}")
        print(f"filepath : {filepath}")
        command_chain = init_command_chain_with_config(nb, ompPlaces)
        measure_command = f"hyperfine --warmup {warmups} --runs {runs} --export-json {filepath} '{program_path}'"
        command_chain.append(measure_command)
        command_chain.execute()
    
    # def run_setup(name: str, nb_on: bool, omp_places: Optional[str], thp_enabled: str, thp_defrag: str):
    #     command_chain = init_command_chain_with_config_thp(nb_on, omp_places, thp_enabled, thp_defrag)
    #     measure_command = f"hyperfine --warmup {warmups} --runs {runs} --export-json {os.path.join(home_directory, name)}.json -n '{name}' '{program_path}'"
    #     command_chain.append(measure_command)
    #     command_chain.execute()
        
    persist_machine_info(home_directory)
    grid_experiment(
        ["nb", "ompPlaces"],
        [
            [("nb-enabled", True), ("nb-disabled", False)],
            [("sockorder", get_sockorder_omp_places()), ("sockets", "sockets"), ("sequential", get_sequential_omp_places()), ("none", None)]
        ],
        run_with_params
    )
    
    
    
def EXPERIMENT_trace_record(program_path: str, numa_balancing: bool, omp_places: Optional[str], thp_enabled: str, thp_defrag: str, warmups: int = 1):
    home_directory = "~/results"
    command_chain = CommandChain([f"cd {home_directory}"])
    command_chain += init_command_chain_with_config_thp(numa_balancing, omp_places, thp_enabled, thp_defrag)
    command_chain += CommandChain(warmups * [program_path])
    measure_cmd = f"trace-cmd record -p function -l do_huge_pmd_numa_page -F {program_path}"
    command_chain.append(measure_cmd)
    command_chain.execute()
    
        
    
    
    
def HELPER_print_estimated_time(program_duration: float, nruns: int, nwarmups: int, none_coeff: float):
    total_runs_not_none = nruns + nwarmups
    total_runs_none = int(none_coeff * nruns) + nwarmups
    seconds = program_duration * (2 * total_runs_not_none + total_runs_none)
    minutes = seconds / 60
    if minutes < 60 :
        print(f"{minutes} minutes")
    else :
        print(f"{minutes / 60} hours")
    
    
    

    


def clean_local_directories():
    get_shell_command_output(f"rm -rf ~/results/*")


# clean_local_directories()

# EXPERIMENT_benchmark_different_config("/root/npb/NPB3.4-OMP/bin/bt.B.x", 50, 2) # 8 * 50 * 8 sec = 1h
# EXPERIMENT_benchmark_different_config("/root/npb/NPB3.4-OMP/bin/cg.B.x", 100, 2) # 8 * 100 * 3 sec = 1h
# EXPERIMENT_benchmark_different_config("/root/npb/NPB3.4-OMP/bin/cg.C.x", 50, 2) # 8 * 100 * 3 sec = 1h
# EXPERIMENT_benchmark_different_config("/root/npb/NPB3.4-OMP/bin/mg.D.x", 20, 2) # 8 * 100 * 3 sec = 3h
# EXPERIMENT_benchmark_different_config("/root/npb/NPB3.4-OMP/bin/sp.C.x", 20, 2) # 8 * 100 * 3 sec = 1.5h










# HELPER_print_estimated_time(15, 20, 2, 2)
# EXPERIMENT_benchmark_perf_l3_miss("/root/npb/NPB3.4-OMP/bin/cg.C.x", 20, 2, 2)

# benchmark_different_config("/root/npb/NPB3.4-OMP/bin/cg.C.x", 2)

# print(f"Estimated time : {(10 * (4 * 202 + 2 * 402)) / 3600} h")
# benchmark_perf_l3_miss("/root/npb/NPB3.4-OMP/bin/bt.C.x", 200, 2, 2)

# print(f"Estimated time : {(40 * (4 * 102 + 2 * 302)) / 3600} h")
# benchmark_perf_l3_miss("/root/npb/NPB3.4-OMP/bin/bt.C.x", 100, 2, 3)

# experiment_memory_allocation_and_acces_in_sequential_pinning("/root/npb/NPB3.4-OMP/bin/cg.C.x")


### Revealed the allocation vs access patterns
# EXPERIMENT_mem_access_alloc_l3_miss("/root/npb/NPB3.4-OMP/bin/cg.C.x", warmups=1, capture_l3_miss_local_dram=False)

# EXPERIMENT_quick_comparison("/root/npb/NPB3.4-OMP/bin/cg.C.x", runs=5)

# EXPERIMENT_trace_record(
#     "/root/npb/NPB3.4-OMP/bin/cg.C.x", 
#     numa_balancing=False, 
#     omp_places=get_sequential_omp_places(), 
#     thp_enabled="always", 
#     thp_defrag="always"
# )

EXPERIMENT_mem_access_alloc_l3_miss("/root/npb/NPB3.4-OMP/bin/cg.C.x", warmups=2)
