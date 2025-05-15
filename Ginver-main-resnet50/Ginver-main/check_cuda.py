# list torich devices
import torch
import logging
from typing import List, Dict, Any, Union
from pathlib import Path

def check_cuda() -> bool:
    """
    Check if CUDA is available and the GPU is supported.
    Returns True if CUDA is available and the GPU is supported, False otherwise.
    """
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return False

        # Get the current device index
        device_index = torch.cuda.current_device()

        # Get the name of the current device
        device_name = torch.cuda.get_device_name(device_index)

        # Check if the device name contains 'Tesla' or 'GeForce'
        if 'Tesla' in device_name or 'GeForce' in device_name:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error checking CUDA: {e}")
        return False

def get_cuda_device() -> Union[torch.device, None]:
    """
    Get the CUDA device if available, otherwise return None.
    Returns:
        torch.device: The CUDA device if available, otherwise None.
    """
    if check_cuda():
        return torch.device("cuda")
    else:
        return None
    
def get_cpu_device() -> torch.device:
    """
    Get the CPU device.
    Returns:
        torch.device: The CPU device.
    """
    return torch.device("cpu")

def get_device() -> torch.device:
    """
    Get the device (CUDA or CPU) based on availability.
    Returns:
        torch.device: The device (CUDA or CPU).
    """
    if check_cuda():
        return get_cuda_device()
    else:
        return get_cpu_device()
    
def get_device_name(device: torch.device) -> str:
    """
    Get the name of the device.
    Args:
        device (torch.device): The device.
    Returns:
        str: The name of the device.
    """
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    else:
        return "CPU"

def get_device_properties(device: torch.device) -> Dict[str, Any]:
    """
    Get the properties of the device.
    Args:
        device (torch.device): The device.
    Returns:
        Dict[str, Any]: The properties of the device.
    """
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(device)
    else:
        return {}
    
def get_device_memory(device: torch.device) -> Dict[str, int]:
    """
    Get the memory of the device.
    Args:
        device (torch.device): The device.
    Returns:
        Dict[str, int]: The memory of the device.
    """
    if device.type == 'cuda':
        return {
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
            "free_memory": torch.cuda.memory_allocated(device),
            "used_memory": torch.cuda.memory_reserved(device)
        }
    else:
        return {}
    
def get_device_info() -> Dict[str, Union[str, Dict[str, Any]]]:
    """
    Get the device information.
    Returns:
        Dict[str, Union[str, Dict[str, Any]]]: The device information.
    """
    device = get_device()
    device_name = get_device_name(device)
    device_properties = get_device_properties(device)
    device_memory = get_device_memory(device)
    
    return {
        "device": str(device),
        "device_name": device_name,
        "device_properties": device_properties,
        "device_memory": device_memory
    }

if __name__ == "__main__":
    device_info = get_device_info()
    print("Device Information:")
    print(f"Device: {device_info['device']}")
    print(f"Device Name: {device_info['device_name']}")
    print(f"Device Properties: {device_info['device_properties']}")
    print(f"Device Memory: {device_info['device_memory']}")
