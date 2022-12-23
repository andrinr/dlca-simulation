# Neural Cellular Automata (NCA) to learn Physics simulations

A project based on: *Growing Neural Cellular Automata* https://distill.pub/2020/growing-ca/

Realized during the course *Physically-Based Simulation in Computer Graphics HS2022* @ETHZ.

Contributors:

Salimbeni Etienne Alain Jaroslav, Andrin Rehmann

## How to reproduce:

All the .ipynb files are also available online via google collab:

- [mainNCA.ipynb](https://colab.research.google.com/drive/1K2Eogp9hiieuxIuShUuhhVIE9bJ264K2?usp=sharing): Most of the code was written here, it includes fluids, soft bodies and also style transfer experiments.
- [dlca_with_sim.ipynb](https://colab.research.google.com/drive/1HPg_dMIGGCdZ8mJWoeTOIaRWQmiKeWQ3?usp=sharing): This version branched of from the main as it uses another approach where the simulation code is integrated into the code to accelerate the runtime. This version achieved the best results to predict simulation steps for incompressible fluids.

If you decide to run the machine learning codes locally, make sure to remove all commands starting with an ``!`` and install the requirements. 

The datasets for ``mainNCA.ipynb`` are simulated locally and acessed by the code over the github servers. 

## Final presentation

The final presentation can be viewed here or be accessed via figma, which is recommended as the GIF animations are not shown in the PDF file: 

[Link to presentation](https://www.figma.com/file/XzlSP8irZH7dJJ0ICHBnaW/dlca?node-id=182%3A2&t=FleN39ftbAMWeFrY-1)
