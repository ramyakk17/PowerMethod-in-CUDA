# Add source files here
EXECUTABLE	:= power
# Cuda source files (compiled with cudacc)
CUFILES_sm_20	:= power.cu 
CUDEPS		:= power_gpu:.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 

################################################################################
# Rules and targets

include ../../common/common.mk
