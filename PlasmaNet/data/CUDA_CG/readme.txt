For launching the solver you need first to install the cpp extension.
To do so, follow the following steps (they work for my personal case, 
but they might not be the most accurate procedure to follow, so I am open
to any correction or suggestion on the subject):

1. purge all your module
2. module load compiler/gcc/8.3.0
3. Check if your cuda path is correct!
Personally I added the following to mi .bashrc

#CUDA 
export CUDA_HOME='/softs/nvidia/cuda-10.1'
export CUDA_PATH='/softs/nvidia/cuda-10.1'
export LD_LIBRARY_PATH='/softs/nvidia/cuda-10.1/lib64/:/softs/nvidia/cuda-10.1/:/softs/nvidia/cuda-10.1/extras/CUPTI/lib64/'

Specially the Cuda path is important!

4. Try installing the plugin:

python3 setup.py install

if problems with CUDA path try:

CFLAGS="-I${CUDA_PATH}/include -I${CUDA_PATH}/samples/common/inc " python3 setup.py install
CPATH=/softs/nvidia/cuda-10.1/include:/softs/nvidia/cuda-10.1/samples/common/inc python3 setup.py install

5. Almost there! The solver has to be launched in CUDA, so you can either launch it with sbatch
or you can allocate some memory on the gpu and ssh to the gpu in order to launch in bash.

salloc -N 1 --ntasks-per-node=12 -p gpu --gres gpu:1 --time=04:00:00

