'''
Shared Block: ResBlock, DownSample, UpSample
'''
from timestep_block import *

class ResBlock(TimestepBlock):
    def __init__(self):
        super().__init__()

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()