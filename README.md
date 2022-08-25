# P1Navigation
Deep Reinforcement Learning Project 1: Banana Collection

This program provides a method for training a Deep Q network to control an agent while it explores an environment and collects yellow bananas without gathering blue bananas. For each yellow banana, a score of +1 is obtained, while collecting a blue banana incurs a penalty of -1 points. The environment is considered solved when an average score of +13 is achieved. The environment details are as follows:

Environment (UnityBrainName): BananaBrain<br>
&nbsp;&nbsp;&nbsp;&nbsp;Observation space type: continuous<br>
&nbsp;&nbsp;&nbsp;&nbsp;Observation space size: 37<br>
&nbsp;&nbsp;&nbsp;&nbsp;Action space type: discrete<br>
&nbsp;&nbsp;&nbsp;&nbsp;Action space size: 4<br>

The current Unity implementation is designed to run on Windows; alternative Unity environments for Linux and iOS are also available, but beyond the scope of the current deployment.

REQUIREMENTS:

A Python 3+ Anaconda Distribution (download from https://www.anaconda.com/products/distribution)

During Anaconda installation, ensure that Python is added to the environment PATH.

RECOMMENDED:

Github Desktop (available from https://desktop.github.com/) 

TO INSTALL:

Create a new conda environment with Python 3.6:<br>
conda create --name {name} python==3.6.15

Activate conda environment:<br>
conda activate {name}

Install old PyTorch version:<br>
conda install pytorch==0.4.0 -c pytorch

Install OpenAI Gym and the box2d environment:<br>
pip install gym<br>
pip install gym[box2d]

Clone Github repository for P1-Navigation project:<br>
git clone https://github.com/udacity/Value-based-methods.git

Navigate to the python directory:<br>
cd python

Install additional dependencies:<br>
pip install .

TO RUN:<br>

Download or clone the solution repository:
git clone https://github.com/TrystanMayfaire/P1-Navigation.git

Replace the p1_navigation folder files in the Value-based-methods<br>
folder with the solution repository files

For Jupyter Notebook:<br>
&nbsp;&nbsp;&nbsp;&nbsp;Navigate to the p1_navigation directory<br>
&nbsp;&nbsp;&nbsp;&nbsp;Run Jupyter Notebook: jupyter notebook<br>
&nbsp;&nbsp;&nbsp;&nbsp;Click on Navigation.ipynb to open the notebook<br>
&nbsp;&nbsp;&nbsp;&nbsp;In the menu bar, click on Cells, then Run All<br>

For console:<br>
&nbsp;&nbsp;&nbsp;&nbsp;Navigate to the p1_navigation directory<br>
&nbsp;&nbsp;&nbsp;&nbsp;Run Navigate.py: python Navigation.py

