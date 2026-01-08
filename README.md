<meta http-equiv="Content-Type" content="text/html; charset=utf-8">

# Machine learning-based prediction of ductility of strain-hardening fiber-reinforced cementitious composites 

### Author:
- **Tan Duy PHAN, Van Thong NGUYEN, Dong Joo KIM**  
  Email: tanduy05081998@gmail.com
  Corresponding author: djkim75@hanyang.ac.kr; Telephone: +82 02 2220 0413
  Department of Civil and Environmental Engineering, Hanyang University, 222, Wangsimni-ro, Seongdong-gu, Seoul 04763, Republic of Korea

AIStructDynSolve is an artificial intelligence (AI) powered framework designed to solve both forward and inverse problems in structural dynamics. 
It leverages advanced artificial intelligence methods - particularly physics-informed neural networks (PINNs), Physics-Informed Kolmogorov-Arnold Network(PIKANs) and their extensions - to model, predict, and analyze dynamic structural responses under various loading scenarios, such as seismic excitations.

### The framework solves the following ODE of MDOF:

- M\*U_dotdot+C\*U_dot+K*U=Pt

- Initial Conditions:
   U(t=0)=InitialU
   U_dot(t=0)=InitialU_dot

### The framework aims to:
- Accurately simulate time-dependent structural behavior (forward problems).
- Identify structural parameters or input forces from measured responses (inverse problems).
- Incorporate domain knowledge and physical laws for improved generalization and interpretability.
- Address challenges in multi-frequency, multi-scale dynamics, especially in earthquake engineering applications.
