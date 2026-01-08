<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

# Machine learning-based prediction of ductility of strain-hardening fiber-reinforced cementitious composites 

### Author:
- **Tan Duy PHAN, Van Thong NGUYEN, Dong Joo KIM**  
  - Email: tanduy05081998@gmail.com.
  - Corresponding author: djkim75@hanyang.ac.kr; Telephone: +82 02 2220 0413.
  - Department of Civil and Environmental Engineering, Hanyang University, 222, Wangsimni-ro, Seongdong-gu, Seoul 04763, Republic of Korea.

### The framework aims to:
- To develop a reliable ML model capable of predicting strain capacity and crack spacing of SH-FRCCs with experimental validation.
- To determine the effects of input parameters on both strain capacity and crack spacing of SH-FRCCs using SHAP and PDP.
- To experimentally validate the developed ML mode.

### Implementation details for Reproducibility
- Python version: 3.10.
- Package versions: numpy == X.Y.Z; pandas == X.Y.Z; scikit-learn == X.Y.Z; matplotlib == X.Y.Z; seaborn == X.Y.Z.
- Random seeds: All random processes used a fixed seed of 42.
- Data were split into 70/30 train/test sets.
- Cross-validation: number of folds: K = 10.
- Bayesian Optimization configuration: Bayesian optimization was performed using scikit-optimize for 100 trials; metric optimized: R-square; Cross-validation inside objective:5 cv; The best configuration was retrained on the full training set and evaluated once on the held-out test set.
