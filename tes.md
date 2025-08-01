!pip install nvidia-ml-py3

import torch
import subprocess
import pynvml

def cek_info_gpu():
    if not torch.cuda.is_available():
        print("❌ GPU tidak tersedia di runtime ini.")
        return
    
    # Info dasar dari PyTorch
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    total_mem = round(torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3), 2)

    print(f"🖥️  GPU Aktif     : {gpu_name}")
    print(f"📦 Total Memori   : {total_mem} GB")

    # Detail alokasi dari PyTorch
    allocated = round(torch.cuda.memory_allocated() / (1024**3), 2)
    reserved = round(torch.cuda.memory_reserved() / (1024**3), 2)

    print(f"📈 Memori Dialokasikan (aktif): {allocated} GB")
    print(f"📦 Memori Direservasi (buffer): {reserved} GB")

    # Info real-time via NVIDIA Management Library (NVML)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        total_real = round(meminfo.total / (1024**3), 2)
        used_real = round(meminfo.used / (1024**3), 2)
        free_real = round(meminfo.free / (1024**3), 2)

        print("\n🔍 Info Memori Real-Time (NVML):")
        print(f"🔹 Total     : {total_real} GB")
        print(f"🔹 Digunakan : {used_real} GB")
        print(f"🔹 Tersisa   : {free_real} GB")
        print(f"⚙️  GPU Utilization: {util.gpu}%")
        pynvml.nvmlShutdown()
    except Exception as e:
        print("⚠️ Gagal mengambil info dari NVML. Error:")
        print(e)

    # Tambahan: nvidia-smi CLI
    try:
        print("\n📊 nvidia-smi (tambahan):\n")
        subprocess.run(['nvidia-smi'], check=True)
    except:
        print("⚠️ nvidia-smi tidak tersedia di lingkungan ini.")
#6rt
cek_info_gpu()