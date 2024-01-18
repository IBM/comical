<!-- ABOUT THE PROJECT -->
## :dna: Comical: Contrastive multi-omics association learning

[![Comical Diagram][comical]](#)

<!-- <a>![badge-alt-text](images/comical.jpg)</a> -->

Due to the sophisticated nature of complex diseases, finding interpretable associations between multi-omics data can be challenging using standard approaches. 

We propose a contrastive learning approach leveraging multi-omics data to generate many-to-many associations between any two types of multi-omics information. We generate learnable embeddings from tokenizations of each modality and utilize attention-based encoders to learn the connections between them. 

Our modal-agnostic approach uniquely identifies many-to-many associations via self-supervised learning schemes and cross-modal attention encoders. Our method also provides a pre-trained model for many-to-many multi-omic association discovery.

### Built With

<!-- This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples. -->

[![Anaconda][Anaconda.com]][Anaconda-url]
[![Python][Python.com]][Python-url]
[![VSCode][VSCode.com]][VSCode-url]
[![Github][Github.com]][Github-url]


<!-- [![Jupyter][Jupyter.com]][Jupyter-url] -->

## Getting Started

### Prerequisites
1. Create the environment from the `comical_env.yml` file:
```
conda env create -f comical_env.yml
```
* Note: if you receive the error `bash: conda: command not found...`, you need to install Anaconda to your development environment (see "Additional resources" below)
2. Activate the new environment:
```
conda activate comical-env
```
3. Verify that the new environment was installed correctly:
```
conda env list
```
* Additional resources:
   * [Connect to computing cluster](http://ccc.pok.ibm.com:1313/gettingstarted/newusers/connecting/)
   * [Set up / install Anaconda on remote linux server](https://kengchichang.com/post/conda-linux/)
   * [Set up remote development environment using VSCode](https://code.visualstudio.com/docs/remote/ssh)

<a name="running_comical"></a>
### Running Comical

<!-- [![Notebook Template][notebook]](#running_comical) -->

1. Request resources from computing cluster:
```
jbsub -cores 2+1 -q x86_1h -mem 800g -interactive bash
```
2. Activate the new environment:
```
conda activate comical-env
```
3. Move to directory with source code and data:
```
cd /dccstor/ukb-pgx/comical/comical
```
4. Run Comical:
```
nohup python wrapper.py --fname_out_root new_run_check_code --epochs 4 --top_n_perc 0.5 &
```

<!-- * Note: must be run from same directory containing `data/` folder with the following dependencies:
  * snp-encodings-from-vcf.csv
  * T1_struct_brainMRI_IDPs.csv
  * T1mri.csv
  * pairs.csv -->

### Help
```
python wrapper.py --help
```

## Authors

Contributors and contact info:

* Diego Machado Reyes
* Myson Burch (myson dot burch at ibm dot com)
* Aritra Bose (a dot bose at ibm dot com)
* Laxmi Parida (parida at us dot ibm dot com)

## Version History

<!-- * 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]() -->
* 0.1
    * Initial Release

<!-- ## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[comical]: images/comical.jpg
[notebook]: images/screenshot.png

[VSCode.com]: https://img.shields.io/badge/Visual_Studio_Code-033b66?style=for-the-badge&logo=visual%20studio%20code&logoColor=white
[VSCode-url]: https://code.visualstudio.com

[Python.com]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org

[Jupyter.com]: https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white 
[Jupyter-url]: https://jupyter.org

[Github.com]: https://img.shields.io/badge/github-%23006567.svg?style=for-the-badge&logo=github&logoColor=white 
[Github-url]: https://github.com

[Anaconda.com]: https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white
[Anaconda-url]: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment

