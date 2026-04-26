# MaSDM

**An Adaptive Knowledge Transfer Probability Matrix based Evolutionary Many Task Optimization Framework**

This repository provides the MATLAB implementation of **MaSDM**, an evolutionary many task optimization framework for single objective many task optimization problems. The framework includes two variants:

- **MaSDM_GA**: a genetic algorithm based variant
- **MaSDM_DE**: a differential evolution based variant

MaSDM is designed around three tightly coupled stages:

1. **Knowledge acquisition** via structural feature based task similarity analysis.
2. **Knowledge transfer** via dual utilization of transfer evaluation, which dynamically builds a transfer reliability matrix.
3. **Knowledge utilization** via two complementary inter task reproduction strategies: elite region projection and isotropic Gaussian model based sampling.

The paper reports that the proposed framework achieves strong performance on the **CEC19** and **WCCI20** many task benchmark suites, and the two released variants are compared against representative GA based and DE based baselines under the same experimental setting.



# How to Use

## 1. Install the MTO Platform

First, download and configure the **MTO Platform (MToP)** https://github.com/intLyc/MTO-Platform.

## 2. Add the algorithm files

Copy the following files into the corresponding algorithm folder of the MTO Platform:

- `MaSDM_GA.m`
- `MaSDM_DE.m`

## 3. Run MaSDM in the platform

You can run the algorithms from the MTO Platform GUI or command line by selecting one of the following algorithm names:

- mto('MaSDM_DE','WCCI20_MaTSO1',30,true,50,false,'MaSDM_DEWCCI20SO1');
- mto('MaSDM_GA','WCCI20_MaTSO1',30,true,50,false,'MaSDM_GAWCCI20SO1');

## 4. Recommended experimental setting

According to the paper, the main experiments use:

- **Population size**: `100`
- **Maximum function evaluations**: `5 × 10^6`

The remaining parameter settings can be adjusted in the platform according to your experimental needs.

## 5. Main parameters in the released code

### MaSDM_GA

- `MuC = 2`
- `MuM = 5`
- `KTN = 5`
- `Delta0 = 0.5`
- `Gap = 5`
- `Lambda0 = 0.5`
- `ParaMin = 0.05`
- `ParaMax = 0.95`
- `eta = 30`
- `split = 50`

### MaSDM_DE

- `F = 0.5`
- `CR = 0.7`
- `KTN = 5`
- `Delta0 = 0.5`
- `Gap = 5`
- `Lambda0 = 0.5`
- `ParaMin = 0.05`
- `ParaMax = 0.95`
- `eta = 30`
- `split = 50`
