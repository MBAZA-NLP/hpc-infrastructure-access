# Tutorial: How to Access the DFKI Computing Infrastructure

Members of the Mbaza Community have the great opportunity to access a high-performance computing (HPC) cluster at the [German Research Center for Artificial Intelligence](https://www.dfki.de/en/web).

## What is this all about?
Know the feeling when you want to train a machine learning model on your computer and training just doesn't finish? One solution is to use much more powerful hardware!

This tutorial describes how to access an HPC computing cluster at DFKI which also offers GPUs.

## Prerequisites

Before you get started with running your code and training models you need to sort out two things:

1. Create an account for the **DFKI computing cluster**.

2. Create an account for the **DFKI VPN** and set up the VPN connection following the [steps described here](media/DFKI%20VPN_overview.pdf).

    ![vpn_setup](media/vpn.png)
    
*It is also a good idea to create a **GitHub** account if you do not already have one. We will need it to version control our code and projects.* 

Please note that the computing infrastructure is only available to Mbaza Community members. Community members can get in touch to receive instructions on how to create these accounts.

## Overview of components

Usually we will work with three components:

- GitHub/Gitlab for version control
- The remote server for storing large files and running jobs on GPUs.
- Our local machine (laptop) for code development and repo set-up.

Below you find an example visualisation how these components work with each other and interact. 

![components_overview](media/components_overview.png)

## Set up your development environments

Before jumping in the fun part - model training - we should take some time to properly set-up and connect the components mentioned above. The following steps are only one way to do it. Feel free to do it as it works best for you!

### Recommended Workflow

1. Create your repo with a consistent folder structure (e. g. using the [Cookiecutter structure](https://drivendata.github.io/cookiecutter-data-science/)) on your local machine.
2. Push this repo to GitHub/Gitlab.
    - IMPORTANT: Include large folders/files in the gitignore. We do not want to push our datasets (if they are large) and trained models to GitHub but keep them only on the remote server!
3. Connect to the remote server ([see below](#connect-to-the-remote-server)) and ```git clone``` the repo from GitHub/Gitlab on the server in the file directory of your choice: This will most likely be in ```/data/USERNAME/```.
4. If the dataset is not on GitHub (for instance if it is larger than 100MB), use ```scp``` to copy the large dataset files from your local machine to the remote server ([see below on how to](#sending-data-and-files-to-remote-server)). Put it, for example, under ```/data/USERNAME/PROJECT_FOLDER```. 
*Advanced: Create a new data, models folder in the cloned repo and ```scp``` the large files directly there. You MUST specify the gitignore accordingly then. Easier to just drop the data in a separate folder on the remote server and mount this folder when training.*

—> Now you have set up the base structure:
- You have a Repo pushed to GitHub for version control. **✓** 
- You have the data and repo on your local machine for code development. **✓** 
- You have the repo on the remote server to pull changes and run the code with the datasets and GPUs. **✓** 

 ### Workflow to stay in sync

1. Develop code in your IDE. 
2. Push new commits from local machine to GitHub. This is the usual workflow without interacting with a remote server for model training. 
3. Keep the repo on the remote server up-to-date by pulling changes from GitHub to remote server. 

*Note:* 
- If there are any large files that should not enter GitHub, ```scp``` them directly from local machine to remote server.
- If raw data has to be transformed and it needs GPU support (or you simply decide to run all jobs on the remote machine which is recommended), run it on the remote server and directly put it in a data folder there.

## Connect to the remote server

Now we can connect to the remote Server. Note that the server runs on Linux, so you should get familiar with simple Linux Terminal commands (see [here](https://linuxconfig.org/bash-scripting-tutorial-for-beginners) for an introduction).

1. Connect to the DFKI VPN.
2. Open terminal on your local machine and ```ssh``` into remote server using your Computing Cluster Credentials (username and password):
    
    ``` bash
    ssh USERNAME@serv-6404.kl.dfki.de
    ```
3. (Optional) Check available hardware resources and information about the cluster:

    | Description |Command|
    |---|---|
    | Check available disk space | ```df -h``` |
    | Check disk usage of current folder | ```du -sch``` |
    | Check disk usage of each sub-directory | ```du -hsx $(ls -A) \| sort -rh``` |
    | Infos about the computing cluster | ```sinfo``` or ```clusterinfo``` |

## Understand the cluster architecture
- Login node
- File system (home, data, enroot)
- Containers
- [Slurm](https://slurm.schedmd.com/)

![giz_cluster_structure](media/giz_cluster_structure.png)

Once you are connected you can take a look at the folder structure:

``` bash
ls -1
```

![container_folder_structure](media/container_folder_structure.png)

## Scheduling and running jobs
- ```srun``` and its parameters

- Job queue and resource allocation (full resources needed, allocated to jobs that fit, not chronologically)

You can also check which jobs are currently running or next in line (**job queue**):
| Description |Command|
|---|---|
| Check queued jobs | ```squeue``` |
| Check your own queued jobs | ```squeue -u <your-user-name>``` |
| Get table of queued jobs | ```squeue -t R -O JobID:8,UserName:12,tres:60``` |

In the table of queued jobs, ```R``` stands for running and ```PD``` for pending.

- usrun.sh user script

To check if GPUs are running:
    
``` bash
usrun.sh --gpus=8 nvidia-smi
```
    
You should see this:

![check_gpus_running](media/check_gpus_running.png)

- Interactive session
- ```scancel <job-id>``` or ctrl-c ctrl-c in output console

### Attach and detach, screen
``` bash
sattach <jobid>.0
```

Screen
| Description | Command |
|---|---|
| Create screen | ```screen``` |
| Detach running screen | ```Ctrl-a Ctrl-d``` |
| Show existing screens | ```screen -ls``` |
| Attach to running screen | ```screen -r``` |

see also: https://linuxize.com/post/how-to-use-linux-screen/

## Set up your execution environment
For Machine Learning workloads, you will need a set of packages such as Pandas, PyTorch, TensorFlow or scikit-learn. Depending on what you need, you have to take different steps to set up your execution environment:
- Ideally, there is already an image available with all packages you need: [Use a ready-made image](#use-a-ready-made-image)
- If you use a Conda ```environment.yml``` file: [See here](#using-conda-environmentyml)
- If you use a pip ```requirements.txt```: [See here](#install-packages-into-the-container-with-pip-requirementstxt)
- If you want to use a Docker image available online: [Use an image from Docker Hub or Nvidia](#use-images-from-docker-hub-or-nvidia)

### Use a ready-made image
To see what pre-installed images contain packages we need we can *grep* them. For example to see all containers that have ```pandas``` pre-installed you can run:

``` bash
grep pandas /data/enroot/*.packages
```

![show_images_with_pandas](media/show_images_with_pandas.png)

Now it is important to understand the interaction between the images, your job and virtual environments. Generally we want to first choose a container and mount it. Whatever we do next is done within this container.
This is important since we need to work inside these containers to ensure proper set-up and utilization of the Nvidia’s GPUs. 
The default command to mount the latest pytorch image and if you created a customized data folder ```test_folder``` to store your datasets would be:

    TODO

Now, we run a slurm ```srun```  command to mount and display all the pre-installed python packages within this container:

```
srun \
  --container-image=/data/enroot/nvcr.io_nvidia_pytorch_22.05-py3.sqsh \
pip3 list
```

![display_python_packages](media/display_python_packages.png)

This is only the first few lines, we can see that there are a lot of preinstalled python packages. In the ideal case all your requirements and dependencies are already installed and you can simply choose a ready-made image.

### Using Conda ```environment.yml```

#### Solution 1: Install environment locally and make it available in container

Miniconda: see https://docs.conda.io/en/latest/miniconda.html#linux-installers

``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
chmod a+x *.sh
./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```

IMPORTANT: Set the installation location to ```/data/USERNAME/miniconda```. By default, conda will install to your ```/home/USERNAME/``` folder, which is not meant for large installations since it is too small.

Create and test Conda environment:

``` bash
conda env create -f /home/steffen/demos/environment.yml
conda activate demo
python -u -c "import torch; print(f'PyTorch {torch.__version__}')"
# prints PyTorch 1.12
```

Activate Conda environment in container:

Can be run on images without any Conda installation,
e.g. use image ```/data/enroot/nvcr.io+nvidia+cuda+11.6.2-cudnn8-runtime-ubuntu20.04.sqsh``` 
```usrun.sh ./demos/local-conda-env-demo.sh```

``` bash
#!/bin/sh
# activate local Conda environment
. /data/steffen/miniconda/etc/profile.d/conda.sh
conda activate demo
python -u -c "import torch; print(f'PyTorch {torch.__version__}')"
# prints PyTorch 1.12
```

#### Solution 2: Install Conda in container and create environment
For example use image ```/data/enroot/nvcr.io+nvidia+cuda+11.6.2-cudnn8-runtime-ubuntu20.04.sqsh``` --> ```usrun.sh ./demos/local-conda-env-demo.sh```

``` bash
#!/bin/bash
apt update
apt install -y wget
cd /usr/local
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
chmod a+x Miniconda3-py38_4.12.0-Linux-x86_64.sh
./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p /usr/local/miniconda
. /usr/local/miniconda/etc/profile.d/conda.sh
conda env create -f /home/steffen/demos/environment.yml
conda activate demo
python -u -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Install packages into the container with ```pip requirements.txt```
In this case, install Python and pip in the container and create environment.

For example, use image ```/data/enroot/nvcr.io+nvidia+cuda+11.6.2-cudnn8-runtime-ubuntu20.04.sqsh``` --> ```usrun.sh ./demos/container-install-demo-pip.sh```.

``` bash
#!/bin/bash
apt update
apt install -y python3 python3-pip
pip install -r /home/steffen/demos/requirements.txt
python3 -u -c "import torch; print(f'PyTorch {torch.__version__}')"
```

**Recommendation**: Try to find an image that contains your main ML framework and only install missing libraries at startup.

### Use images from Docker Hub or Nvidia
Docker images can be imported from Docker Hub (https://hub.docker.com/) or Nvidia (https://ngc.nvidia.com/catalog/containers).

#### Example: Import Alpine image to Enroot from *Docker Hub*
Original Docker command (DON'T RUN THIS):
``` bash
docker pull alpine:latest
```

Enroot import command:
``` bash
enroot import docker://alpine:latest
```

#### Example: Import Cuda image from *Nvidia Catalog*
Original Docker command (DON'T RUN THIS):
    
    docker pull nvcr.io/nvidia/cuda:11.2.1-base-ubuntu20.04

Enroot import command:
replace the first / with # from enroot import

    enroot import docker://nvcr.io#nvidia/cuda:11.2.1-base-ubuntu20.04

**Attention 1!** This creates large files in the cache (```$HOME/.cache/enroot```). Clean up afterwards!

**Attention 2!** Using Nvidia requires an Nvidia account and an API key!
API key can be generated here:
https://ngc.nvidia.com/setup/api-key.
To configure Enroot for using your API key, create
```enroot/.credentials``` within your ```$HOME``` and
append the following line to it:

    machine nvcr.io login $oauthtoken password <API_KEY>

### Save the new image from Container
Once you the image in the container is ready and you have made all changes you wanted, you can save the image permanently to be able to reuse it later:

``` bash
usrun.sh --container-save=/data/steffen/my-image.sqsh --pty bash
```

## Sending data and files to remote server
Another important requirement for running Machine Learning workloads is to have the training and test data readily available on the server. 

We want to store our large raw and processed datasets as well as trained models only on the remote server (remember, that is why we need to specify the gitignore). But first we need to send it the remote server. For this we use [```scp```](https://www.ionos.com/digitalguide/server/configuration/linux-scp-command/).

We want to use our assigned data folders which can be found under:

``` bash
cd data/USERNAME
```

then we may create a sub-folder for datasets such as

``` bash
mkdir test_data_folder
```
(use ```rmdir``` or ```rm -rf``` for removal)

Then to transfer data from your local machine you must open the *terminal* on your local machine, We may simply use ```scp``` to send secure copies to the servers. The command is the following:

``` bash
scp -r local_file_path ssh destination
```

the destination is the DFKI server and the file path you want to specify, for example using the folder created above it would be:

``` bash
ssh USERNAME@serv-6404.kl.dfki.de:/data/USERNAME/test_data_folder/
```

and the full command would be

``` bash
scp -r local_file_path ssh USERNAME@serv-6404.kl.dfki.de:/data/USERNAME/test_data_folder/
```


## Working example
Now we can do what we came for: Running our code on the remote server and utilising the GPUs. Let’s use this repo as an example:

[https://GitHub.com/jonas-nothnagel/sdg_text_classification](https://GitHub.com/jonas-nothnagel/sdg_text_classification) 

As explained above, connect to the remote server and ```git clone``` the repo into ```/data/USERNAME/```.

First, let’s simply compile a python script without GPU support.  Again, mount the container of your choice, but also specify where the Repository lies on the remote server. Since this is the place where we pushed all the code beforehand: here for example “sdg_text_classification”.

We choose the newest pytorch container (```/data/enroot/nvcr.io_nvidia_pytorch_22.05-py3.sqsh```) and run our training script (```./src/train.py```) using 8 GPUs:

```bash
srun -K --gpus=8 -p batch  \
--container-workdir=`pwd`  \
--container-mounts=/data/nothnagel/sdg_text_classification:/data/nothnagel/sdg_text_classification  \
--container-image=/data/enroot/nvcr.io_nvidia_pytorch_22.05-py3.sqsh  \
python ./src/train.py
```

It is possible that you run into an error here:

    ERROR: The GPUs have to be specified correctly still.

If this happens, consult [http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/docs/slurm-cluster/resource-allocation/](http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/docs/slurm-cluster/resource-allocation/).

### Bashing

It is good practice to not copy paste these code lines into the terminal directly but to write a **[bash script](https://GitHub.com/jonas-nothnagel/sdg_text_classification/blob/main/run_example.sh)** and compile it with ```bash run_example.sh```.

```bash
#!/bin/bash
srun \
  --container-image=/data/enroot/nvcr.io_nvidia_pytorch_22.05-py3.sqsh \
  --container-workdir=`pwd` \
  --container-mounts=/data/nothnagel/sdg_text_classification:/data/nothnagel/sdg_text_classification \
  python ./src/test.py
```

We want to do this because in the bash script we can specify the whole job, including:

- Creating/activating a virtual environment if necessary
- installing additional dependencies
- specifying the GPU support
- specifying experiment tracking and where to put results etc.

## Further reading
- [DFKI HPC Cluster documentation](http://projects.dfki.uni-kl.de/km-publications/web/ML/core/hpc-doc/)
    - Careful: Not everything applies.