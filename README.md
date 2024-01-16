# One-Shot Heterogeneous Transfer Learning from Calculated Crystal Structures to Experimentally Observed Materials

## Abstract
Data-driven methods in materials science typically suffer from a lack of experimentally collected training data.
Although various transfer learning methods based on source materials datasets containing calculated crystal structures have been widely studied to overcome the lack of training data, existing methods have two fundamental limitations.
(1) They assume the same modality of input material descriptors in source and target datasets, but crystal structures of the materials are not available in most experimental data.
(2) They should re-train all source models on the target dataset to select an appropriate source model.
To overcome these limitations, we propose a modality-free transfer learning method to estimate the effectiveness of the transferred source model without re-training on the target dataset.
The estimated transferabilities showed a strong negative correlation of -0.744 with the training losses of the transferred source models, and we were able to select appropriate source models 105-156 times faster than existing methods.


## Run
- train_src_model.py: A script to train the source models.
- calc_transferability.py: A script to calculate the transferability based on SCOTL.


## Datasets
The GWBC and HOIP-HSE datasets, which were used as the source calculation datasets, are available in [1] and [2], respectively.
The lists of the materials in the MPS and MPL datasets are presented in [the public repository of CGCNN](https://github.com/txie-93/cgcnn).
The crystal structures of all materials in the source calculations datasets are available at the Materials Project database [3].
All target experimental datasets are publicly available in their original papers, as presented in Table 1 of the paper.

- Lee, J., Seko, A., Shitara, K., Nakayama, K., & Tanaka, I. (2016). Prediction model of band gap for inorganic compounds by combination of density functional theory calculations and machine learning techniques. Physical Review B, 93(11), 115104.


## References
[1] Lee, J., Seko, A., Shitara, K., Nakayama, K., & Tanaka, I. (2016). Prediction model of band gap for inorganic compounds by combination of density functional theory calculations and machine learning techniques. Physical Review B, 93(11), 115104.

[2] Kim, C., Huan, T. D., Krishnan, S., & Ramprasad, R. (2017). A hybrid organic-inorganic perovskite dataset. Scientific Data, 4(1), 1-11.

[3] Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL materials, 1(1).
