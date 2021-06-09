import subprocess, os
import matplotlib.pyplot as plt
import pandas as pd

def get_performance(cmd):
    command = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    return command.stdout.decode('utf-8').split()[-1]

cmd = "./model/{} ../sample_inputs/{} {} {} {} {} {}"

def search_sync():
    file_name = "glife_kernel_1"
    input_name = "make-a_71_81"
    method_names = ["single_thread", "multi_thread", "gpu"]
    display = 0
    cores= [1, 16, 0]
    gens = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    result = {}

    result['gens'] = gens

    for method_name, core in zip(method_names, cores):
        print(method_name+'...', end="", flush=True)
        width, height = 500, 500
        result[method_name] = []
        for gen in gens:
            cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
            result[method_name].append(get_performance(cmd_exc))
        print('done')

    df = pd.DataFrame(result)
    df.to_csv("gen.csv")

def search_size():
    file_name = "glife_kernel_1"
    input_name = "make-a_71_81"
    method_names = ["single_thread", "multi_thread", "gpu"]
    display = 0
    cores= [1, 16, 0]
    gen = 100
    widths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] 
    heights = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    result = {}

    result['size'] = widths

    for method_name, core in zip(method_names, cores):
        print(method_name+'...', end="", flush=True)
        result[method_name] = []
        for width, height in zip(widths, heights):
            cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
            result[method_name].append(get_performance(cmd_exc))
        print('done')

    df = pd.DataFrame(result)
    df.to_csv("size.csv")

def search_tlp():
    file_name = "glife_kernel_1"
    input_name = "make-a_71_81"
    method_name = "multi_thread"
    display = 0
    gen, width, height = 100, 500, 500
    cores = list(range(1, 17))
    result = {}
    result["cores"] = cores
    result["time"] = []

    print(method_name+'...', end="", flush=True)
    for core in cores:
        cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
        result["time"].append(get_performance(cmd_exc))

    df = pd.DataFrame(result)
    df.to_csv("tlp.csv")
    print('done')
    print(df.to_string())

def search_dlp1():
    file_names = ["glife_kernel_1", "glife_kernel_1_16"]
    input_name = "make-a_71_81"
    method_name = "GPU) thread_size 16 vs 32..."
    display = 0
    core = 0
    gens = [100, 200, 500, 1000, 1000]
    widths = [100, 200, 500, 1000, 10000]
    heights = [100, 200, 500, 1000, 10000]
    result = { "gen": gens, "width": widths, "height": heights }
    print(method_name, end="", flush=True)

    for file_name in file_names:
        result[file_name] = []
        for gen, width, height in zip(gens, widths, heights):
            cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
            result[file_name].append(get_performance(cmd_exc))

    df = pd.DataFrame(result)
    df.to_csv("gpu_thread_size.csv")
    print("done")
    print(df.to_string())

def search_dlp2():
    file_names = ["glife_kernel_1", "glife_kernel_2"]
    input_name = "make-a_71_81"
    method_name = "GPU) difference between shared..."
    display = 0
    core = 0
    gens = [100, 200, 500, 1000, 1000]
    widths = [100, 200, 500, 1000, 10000]
    heights = [100, 200, 500, 1000, 10000]
    result = { "gen": gens, "width": widths, "height": heights }
    print(method_name, end="", flush=True)

    for file_name in file_names:
        result[file_name] = []
        for gen, width, height in zip(gens, widths, heights):
            cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
            result[file_name].append(get_performance(cmd_exc))

    df = pd.DataFrame(result)
    df.to_csv("gpu_shared.csv")
    print("done")
    print(df.to_string())

def search_dlp3():
    file_names = ["glife_kernel_1", "glife_kernel_1_shape"]
    input_name = "shape_input"
    method_name = "GPU) difference between block shape..."
    display = 0
    core = 0
    gens = [100, 200, 500, 1000, 1000]
    widths = [100, 200, 500, 1000, 10000]
    heights = [100, 50, 30, 20, 10]
    result = { "gen": gens, "width": widths, "height": heights }
    print(method_name, end="", flush=True)

    for file_name in file_names:
        result[file_name] = []
        for gen, width, height in zip(gens, widths, heights):
            cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
            result[file_name].append(get_performance(cmd_exc))

    df = pd.DataFrame(result)
    df.to_csv("gpu_shape.csv")
    print("done")
    print(df.to_string())

def search_sync_size_gpu():
    comp_ways = ["bigger size", "bigger iteration"]
    file_name = "glife_kernel_1"
    input_name = "shape_input"
    method_name = "GPU) difference between sync and size..."
    display = 0
    core = 0
    gens = [100, 10000]
    widths = [1000, 100]
    heights = [1000, 100]
    result = { "gen": gens, "width": widths, "height": heights, "performance": [] }
    print(method_name, end="", flush=True)

    
    for way, gen, width, height in zip(comp_ways, gens, widths, heights):
        cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
        result["performance"].append(get_performance(cmd_exc))

    df = pd.DataFrame(result)
    df.to_csv("gpu_comp.csv")
    print("done")
    print(df.to_string())

def search_sync_size_thread():
    comp_ways = ["bigger size", "bigger iteration"]
    file_name = "glife_kernel_1"
    input_name = "shape_input"
    method_name = "Thread) difference between sync and size..."
    display = 0
    core = 16
    gens = [100, 10000]
    widths = [1000, 100]
    heights = [1000, 100]
    result = { "gen": gens, "width": widths, "height": heights, "performance": [] }
    print(method_name, end="", flush=True)

    
    for way, gen, width, height in zip(comp_ways, gens, widths, heights):
        cmd_exc = cmd.format(file_name, input_name, display, core, gen, width, height)
        result["performance"].append(get_performance(cmd_exc))

    df = pd.DataFrame(result)
    df.to_csv("gpu_comp.csv")
    print("done")
    print(df.to_string())


if __name__ == '__main__':
    # search_sync()
    # search_size()
    # search_tlp()
    # search_dlp1()
    # search_dlp2()
    # search_dlp3()
    search_sync_size_gpu()
    search_sync_size_thread()