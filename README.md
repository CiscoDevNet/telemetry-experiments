# ai-diagnosis-notebooks
Experiments to use AI/ML methods to diagnose network and system issues.

[Highlevel Pipeline Notebooks](./highlevel-pipelines/README.md): These notebooks walkthrough the high-level pipeline steps to diagnose MDT and SysLog files.  The notebooks seek to articulate the pipeline sequence to together with the correlating inputs and resulting output of each step. 

[Technique Notebooks](./techniques/README.md): These notebooks walkthrough the techniques and approaches adopted by each step in the correlating [Highlevel Pipelines](../highlevel-pipelines/README.md).

---

## License

This project is licensed to you under the terms of the [Cisco Sample Code License](./LICENSE).

---
## Jupyter Notebook Setup
We recommend creating a specific conda env to host the Jupyter Kernel and correlating notebook dependencies.

[Instructions to install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Once conda is install, the ./env-setup/conda.yaml can be used to create the desired env for these notebooks.

```
conda env create -f /<path to git clone>/ai-diagnosis-notebooks/env-setup/conda_env.yaml
```

This will create a conda env entitled 'ai-diagnosis', this contains a python 3.10 environment with all the dependencies required for the Jupyter Notebook kernel and the ai-diagnosis notebooks.

### AWS Sagemaker Setup

Sagemaker is built on top of the EC2 service, EC2 is not to be considered as persistent storage. The only folder which will stay intact after a restart is `/home/ec2-user/SageMaker` (this is attached to an EBS volume).

By default, conda environments are installed in `/home/ec2-user/anaconda3/envs`, therefore any added custom environments will be lost when the SageMaker instance restarts.  It is possible to create a script to install a custom environment within the [Lifecycle Configuration Script](https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-lifecycle-config.html).  However, these scripts will timeout after 5 minutes and the conda environment may not be completed within this time frame.  

The following procedure should be adopted in SageMaker via the instance terminal, assuming the content of this repository has been uploaded to the `~/SageMaker` path on the instance.
```
$ mkdir -p ~/SageMaker/conda/ai-diagnosis
$ cd ~/SageMaker/conda/ai-diagnosis
$ conda env create -v -p `pwd`/ai-diagnosis --file ~/SageMaker/env-setup/conda_env.yaml
$ source ~/anaconda3/etc/profile.d/conda.sh
$ conda activate /home/ec2-user/SageMaker/conda/ai-diagnosis
$ python -m ipykernel install --user --name ai-diagnosis --display-name "ai-diagnosis"
```

When you load an ai-diagnosis notebook, the ai-diagnosis kernel should be available for selection.