# S3Map
This is the code for fast spherical mapping of cortical surfaces using [S3Map algorithm](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_16).
![Figure framework](https://github.com/BRAIN-Lab-UNC/S3Map/blob/main/examples/fig_framework.png)

# Usage
1. Download or clone this repository into a local folder
2. Open a terminal and run the follwing code (better do this in a conda environment):
```
conda env create -f environment.yml
```
if only cpu is available, you can install the cpu version torch
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
otherwise, install the correct [torch](https://pytorch.org/get-started/locally/) version of your machine

3. Prepare your data, i.e., inner surfaces in vtk format (the file name should end in '.vtk')

4. Simply run "python s3all.py  -h" for the whole pipeline, and the expected output should be.
```
python s3all.py  -h
usage: s3all.py [-h] --input INPUT [--save_interim_results {True,False}]

Superfast spherical surface pipeline 
1. Computing mean curvature and area 
2. Inflating the inner surface and computing sulc 
3. Initial spherical mapping 
4. Distortion correction for initial spherical surface.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        the input inner surface (white matter surface) in vtk format, containing vertices and faces. There should be 'lh' or 'rh' in the filename for identifying left hemisphere or right hemisphere.
  --save_interim_results {True,False}
                        save intermediate results or not, if Ture, there will be many results generated and need more storage

```
An example output log can be found at [here](https://github.com/BRAIN-Lab-UNC/S3Map/blob/main/examples/example_log).

5. Use [paraview](https://www.paraview.org/) to visualize all generated .vtk surfaces, or [read_vtk](https://github.com/zhaofenqiang/S3Map/blob/a96c103f66db443ba52cdafee28af798a527fc54/sphericalunet/utils/vtk.py#L26) into python environment for further processing.

## Train a new model on a new dataset
After data prepration, modify the train.py file to match the training data in your own path. Then, run:
```
python s3map_train.py
```



