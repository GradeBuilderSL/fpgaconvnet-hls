from dataclasses import dataclass
from typing import List

@dataclass
class GenerateWeights:
    name: str
    wr: bool = False

    def __post_init__(self):
        self.type       = "static" if self.wr else "const static"
        self.bram_type  = "RAM" if self.wr else "ROM"

    def generate_def(self):
        return f"""
{self.type} {self.name}_weight_t {self.name}_weights[{self.name.upper()}_COARSE_IN*{self.name.upper()}_COARSE_GROUP][{self.name.upper()}_COARSE_OUT][DIVIDE({self.name.upper()}_WEIGHTS,{self.name.upper()}_COARSE_IN*{self.name.upper()}_COARSE_GROUP*{self.name.upper()}_COARSE_OUT*{self.name.upper()}_KERNEL_SIZE_X*{self.name.upper()}_KERNEL_SIZE_Y)][{self.name.upper()}_KERNEL_SIZE_X][{self.name.upper()}_KERNEL_SIZE_Y] = {{
#include "{self.name}_weights_0.csv"
}};
        """

    def generate_init(self):
        storage_type = "ram_2p" if self.wr else "rom_np"
        return f"""
#pragma HLS ARRAY_PARTITION variable={self.name}_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable={self.name}_weights complete dim=2
#pragma HLS BIND_STORAGE variable={self.name}_weights type={storage_type}
#pragma HLS STABLE variable={self.name}_weights
        """

    def generate_process_param(self):
        N = self.name.upper()
        return (f"{self.name}_weight_t {self.name}_weights"
                f"[{N}_COARSE_IN*{N}_COARSE_GROUP]"
                f"[{N}_COARSE_OUT]"
                f"[DIVIDE({N}_WEIGHTS,{N}_COARSE_IN*{N}_COARSE_GROUP*{N}_COARSE_OUT*{N}_KERNEL_SIZE_X*{N}_KERNEL_SIZE_Y)]"
                f"[{N}_KERNEL_SIZE_X][{N}_KERNEL_SIZE_Y]")

    def __repr__(self):
        return self.__generate_def() + "\n" + self.generate_init()

@dataclass
class GenerateStreams:
    name: str
    type: str
    dims: List[str]

    def __post_init__(self):
        self.stream_dims = "][".join(self.dims)

    def generate_stream(self):
        return f"""
    stream_t({self.type}) {self.name}[{self.stream_dims}];
#pragma HLS STREAM variable={self.name}
#pragma HLS ARRAY_PARTITION variable={self.name} complete dim=0
        """

    def __repr__(self):
        return self.__generate_stream()

@dataclass
class GenerateBiases:
    name: str
    wr_factor: int = 1

    def generate_def(self):
        if self.wr_factor > 1:
            return f"""
const static {self.name}_biases_t {self.name}_biases[{self.wr_factor}][{self.name.upper()}_COARSE_OUT][DIVIDE({self.name.upper()}_FILTERS,{self.name.upper()}_COARSE_OUT)] = {{
#include "{self.name}_biases.csv"
}};
        """
        return f"""
const static {self.name}_biases_t {self.name}_biases[{self.name.upper()}_COARSE_OUT][DIVIDE({self.name.upper()}_FILTERS,{self.name.upper()}_COARSE_OUT)] = {{
#include "{self.name}_biases.csv"
}};
        """

    def generate_init(self):
        return f"""
#pragma HLS ARRAY_PARTITION variable={self.name}_biases complete dim=1
#pragma HLS BIND_STORAGE variable={self.name}_biases type=rom_np
#pragma HLS STABLE variable={self.name}_biases
        """

    def generate_process_param(self):
        N = self.name.upper()
        if self.wr_factor > 1:
            return (f"const {self.name}_biases_t {self.name}_biases"
                    f"[{self.wr_factor}][{N}_COARSE_OUT][DIVIDE({N}_FILTERS,{N}_COARSE_OUT)]")
        return (f"const {self.name}_biases_t {self.name}_biases"
                f"[{N}_COARSE_OUT][DIVIDE({N}_FILTERS,{N}_COARSE_OUT)]")

    def __repr__(self):
        return self.__generate_def() + "\n" + self.generate_init()


