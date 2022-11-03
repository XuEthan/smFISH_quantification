Requirements: 
gcc & g++ version >= 5.4 
anaconda
pip 

How to run demo: 
1. Download the repository 
2. conda create -n demofish python=3.9 
3. conda activate demofish 
4. conda activate cpufish
5. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
6. python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
7. pip install -r requirements.txt
8. python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
9. Navigate to /smFISH_quantification/smFISH_scripts
10. ./GUI_smFISH.py

The output will be in smFISH_quantification/smFISH_parameters/spot_count
