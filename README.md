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

### Implementation details for reproducibility
- Python version: 3.10.
- Package versions: numpy == 2.2.6; pandas == 2.3.3; scikit-learn == 1.7.2; matplotlib == 3.10.7; seaborn == 0.13.2; shap == 0.49.1; PDPbox == 0.2.0.
- Random seeds: All random processes used a fixed seed of 42.
- Data were split into 70/30 train/test sets.
- Cross-validation: number of folds: K = 10.
- Bayesian Optimization configuration: Bayesian optimization was performed using scikit-optimize for 100 trials; metric optimized: R-square; cross-validation inside objective: 5cv; the best configuration was retrained on the full training set and evaluated once on the held-out test set.
