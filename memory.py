import torch
from typing import Optional

class GPUMemoryReserver:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize GPU memory reserver

        Args:
            device: Specify device, e.g., 'cuda:0', defaults to current device
        """
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reserved_tensor = None
        self.reserved_size = 0
        
    def reserve(self, mb: int) -> None:
        """
        Reserve specified amount of GPU memory (MB)

        Args:
            mb: Amount of memory to reserve, in MB
        """
        print(f"device:{self.device}")
        if self.device == 'cpu':
            print("GPUMemoryReserver: Warning: Trying to reserve GPU memory but running on CPU")
            return
            
        if self.reserved_tensor is not None:
            self.release()
            
        bytes_to_reserve = mb * 1024 * 1024  # Convert to bytes
        
        try:
            # Create a large enough tensor to occupy the specified amount of GPU memory
            self.reserved_tensor = torch.empty(
                (bytes_to_reserve // 4,),  # Divide by 4 because float32 occupies 4 bytes
                dtype=torch.float32,
                device=self.device
            )
            self.reserved_size = mb
            print(f"GPUMemoryReserver: Successfully reserved {mb} MB GPU memory on {self.device}")
        except RuntimeError as e:
            print(f"GPUMemoryReserver: Failed to reserve {mb} MB GPU memory: {str(e)}")
            self.reserved_tensor = None
            self.reserved_size = 0
            
    def release(self) -> None:
        """Release reserved GPU memory"""
        if self.reserved_tensor is not None:
            del self.reserved_tensor
            torch.cuda.empty_cache()  # Clear cache
            self.reserved_tensor = None
            self.reserved_size = 0
            print(f"GPUMemoryReserver: Released reserved GPU memory on {self.device}")