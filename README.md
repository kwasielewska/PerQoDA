# PerQoDA
Dataset Quality Assessment with Permutation Testing

Contact person: Katarzyna Wasielewska, email: k.wasielewska@ugr.es

Last update: 29.09.2022

<hr>

## Description

The PerQoDA software is designed to test the quality of a dataset for classification tasks using permutation testing. The method requires a labeled dataset. The intuition is that in a good quality dataset, the association between predictors and the labelling should be strong and unique. Thus, everytime we permute the labels in a good quality dataset, a singificant reduction in the quality is expected. A main innovation is that we permute the labelling at different percentages, so that we can generate a grey scale of quality between bad datasets and good datasets.

The software includes a variety of supervised ML techniques and performance metrics.

### Output examples

To assess the quality of a dataset, this package produces a number of useful visualizations:

- The permutation chart. 

Inspired by previous research on permutation testing (Lindgren et al. Journal of Chemometrics (1996) 10:521), this plot compares true vs permuted performance results in the y-axis while in the x-axis each permutation is located depending on the correlation between the original labelling and the permuted one. That way, resamples of 50\% of permuted labels are located around the 0.5 correlation, while those for 1\% are close to a correlation of 1. The true performance values obtained for each classifier are shown by diamonds. In principle, diamonds should be located at correlation level 1 in the abscissas, since the labelling is not permuted. However, for visualization purposes, we decided to locate them slightly separated at the right, also separated from permuted results. Each diamond is also attached to the horizontal line representing its corresponding (true) performance. The results of the performance calculated after each permutation are shown by circles. Circles with dark border represent permuted models with higher performance than the true one (diamonds).  The lowest performance of all permutes at different percentage levels is marked with a red dashed horizontal line. This provides a baseline of randomness, which can sometimes be unexpectedly high. 

In the example, techniques like Adap or RF show several partially permuted models (at 5\%-10\% permutation) above the quality of the corresponding true model. However, we do find other techniques like KNN that attain a good performance which is also unique, in the sense that no permuted version of the model can attain it.  
 
<img src="https://user-images.githubusercontent.com/80593278/189530888-8c84dadd-ca49-42ab-a040-4208c3e092d1.PNG" width="400">

- The p-value table

The permutation chart is not an optimal visualization for low-percentage permutes, located close to correlation 1. For this reason, we complement this visualization with the p-value table. P-values are also specially useful in Big datasets, where permutation percentages may go very low (<<1\%).  

The table shows that some classifiers reject the null hypothesis (p-values < 0.01) at all permutation levels. This means that there is a strong relationship between the observations and the true labels in the dataset, and any change to the labels tend to degrade the relationship. 

<img src="https://user-images.githubusercontent.com/80593278/189530895-880d6592-5ca1-4ac4-855a-f7ed384035f1.PNG" width="400">

- The slope

The permutation chart and the p-value table are complementary and fully informative to assess the quality of a dataset. Unfortunately, they provide too much information if the goal is to compare the quality of two datasets, for instance, if we want to assess incremental improvements in the quality of the data. To provide a simpler quantification of dataset quality for differential comparisons, we define the permutation slope, which corresponds to the slope of the regression line fitted to the points representing the performance results of the selected classifier (obtained after permutations) at different permutation levels. We obtain one slope per classifier, the largest of which is a measure of the dataset quality (in this case, slope = 0.7955).

<img src="https://user-images.githubusercontent.com/80593278/189530898-9039bbc5-c434-44af-98da-810e5bf08b5a.PNG" width="400">

## Content and instructions

### Python

Folder with python library: python

#### Instalation 
- Install Weles according to the instruction described below (weles.zip is available in this repo) 
- Install modules localed in perqoda.py - pip3 install -r requirements.txt

#### How to run
- Prepare your dataset (feature dataset)
- Edit configuration file
- Run the tool -> python3 perqoda.py

### Jupyter
Folder with Jupyter scripts: jupyter

Items:
- PerQoDA.ipynb - core code
- Examples: Example of dataset quality assessment with PerQoDA and the sample of the inSDN dataset

#### Instalation 
- Install Weles:
1. Unpack Weles.zip on local disk
2. Install the Weles from local folder, for example: pip install -e D:\weles-master, or unzip it the python/jupyter directory for local import of Weles module
3. In Jupyter Notebook, execute the following commands:
   - import sys
   - sys.path.append('D:\weles-master')
and check that the path has been added to the PATH
   - print(sys.path) 
- Install modules localed in the notebook

#### How to run
- Load your dataset (feature dataset)
- Select classifiers and performance metric
- Set the number of permutations and define the permutation policy (percentages of labels)
- Run notebook


## Papers

Camacho, J., Wasielewska, K., Dataset Quality Assessment in Autonomous Networks with Permutation Testing, IEEE/IFIP Network Operations and Management Symposium (NOMS), 2022, pp. 1-4, doi: 10.1109/NOMS54207.2022.9789767

Wasielewska K., Soukup D., Čejka T., Camacho J., Dataset Quality Assessment with Permutation Testing Showcased on Network Traffic Datasets [TechRxiv' 22](https://www.techrxiv.org/articles/preprint/Dataset_Quality_Assessment_with_Permutation_Testing_Showcased_on_Network_Traffic_Datasets/20145539) 

## More info

We used the Weles tool published at https://github.com/w4k2/weles. Data shuffling requires a slight modification of the original code by adding a protocol that supports shuffling methods (look at the 'python' folder).

## Copyright Note

Copyright (C) 2022  Universidad de Granada
Copyright (C) 2022  José Camacho 
Copyright (C) 2022  Katarzyna Wasielewska
Copyright (C) 2022  Dominik Soukup
 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
