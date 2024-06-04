# Maximum Likelihood Weight Estimation (MLWE)

This repository provides the Pytorch code for the work "Maximum Likelihood Weight Estimation for Partial Domain Adaptation" published in Information Sciences, 2024. The video for introducing this work is available at Bilibili with the link https://b23.tv/u9XyLM7.


<video id="video" controls="" preload="none" poster="封面">
      <source id="mp4" src="https://b23.tv/u9XyLM7" type="video/mp4">
</videos>


In this work, we address the Partial Domain Adaptation (PDA) problem. The problem aims to generalize a classification model to an unlabeled target domain by harnessing a related labeled source domain, where the source label space contains the target label space. Two primary challenges in PDA weaken the model's classification performance in the target domain: (i) the joint distribution of the source domain is related but distinct from that of the target domain, and (ii) the source outlier data, whose labels do not belong to the target label space, have a negative impact on learning the target classification model. To tackle these challenges, we propose a Maximum Likelihood Weight Estimation (MLWE) approach to learn a weight function for the source domain. The weight function matches the joint source distribution of the relevant part to the joint target distribution, and mitigates the negative impact of the source outlier data effectively. Specifically, we employ a maximum likelihood method to estimate the weight function. The estimation leads to a convex optimization problem which has a global optimal solution. 


#### Dataset folder
The folder structure required (e.g OfficeHome)
- data
  - OfficeHome
    - list
      - Art_25.txt
      - Art.txt
      - Clipart_25.txt
      - Clipart.txt
      - Product_25.txt
      - Product.txt
      - Real_25.txt
      - Real.txt
    - Art
    - Clipart
    - Product
    - Real

##### How to run

```bash
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Art --target Clipart --seed 0 | tee PDA-OfficeHome_A2C_seed0.log
```


For more details of this partial domain adaptation approach,  please refer to the following work:

@article{Wen2024Maximum,    
title = {Maximum Likelihood Weight Estimation for Partial Domain Adaptation},    
journal = {Information Sciences},    
volume = {676},    
pages = {120800},    
year = {2024},   
url = {https://www.sciencedirect.com/science/article/pii/S002002552400714X},   
author = {Lisheng Wen and Sentao Chen and Zijie Hong and Lin Zheng}   
}

  
The Pytorch code is currently maintained by Lisheng Wen. If you have any questions regarding the code, please contact Lisheng Wen via the email lishengwenmail@126.com.

