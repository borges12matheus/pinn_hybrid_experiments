# Physics-Informed Correction of RANS-Based CFD Fields for Backward-Facing Step Flow Using Hybrid Neural Networks

Accurate prediction of turbulent flow fields remains a significant challenge in Computational Fluid Dynamics (CFD), particularly when high-fidelity simulations are computationally expensive. Data-driven surrogate models provide an attractive alternative but often fail to preserve the governing physical laws, resulting in physically inconsistent predictions. This work presents a hybrid machine learning framework for physics-guided correction of Reynolds-Averaged Navier–Stokes (RANS) solutions for the classical two-dimensional Backward-Facing Step (BFS) benchmark.

The proposed methodology learns correction mappings between coarse- and fine-mesh CFD solutions generated with OpenFOAM. A Multilayer Perceptron (MLP) is first employed as a purely data-driven baseline to estimate velocity and pressure corrections. Subsequently, a Physics-Informed Neural Network (PINN) introduces physics-based regularisation by enforcing the continuity equation and, in ongoing developments, the steady momentum equations. Rather than replacing the CFD solver, the proposed framework acts as a physics-guided corrector that aims to improve numerical predictions while enforcing physical consistency.

Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Relative Error Reduction (RER), and residual-based physical consistency metrics. Preliminary experiments indicate that data-driven models effectively reduce interpolation errors between CFD meshes, whereas physics-informed constraints improve physical consistency while revealing the trade-off between predictive accuracy and conservation-law enforcement. These findings provide valuable insights into the development of hybrid surrogate models for turbulent flow reconstruction.

Furthermore, the proposed methodology supports computationally efficient CFD workflows by combining data-driven learning with embedded physical constraints, enabling more reliable surrogate models for engineering flow prediction.

The present study provides a foundation for future research involving turbulence-aware residuals, adaptive loss weighting, additional turbulence closure information, and validation using higher-fidelity benchmark CFD datasets.

Keywords: Physics-Informed Neural Networks; Computational Fluid Dynamics; Backward-Facing Step; RANS; Surrogate Modelling.

--

# Learning Mechanical Dynamical Systems with Neural Ordinary Differential Equations: A Comparative Study with Classical Numerical Solvers

Accurate modelling of continuous mechanical dynamical systems is fundamental to computational engineering, where ordinary differential equations (ODEs) describe the evolution of structural and vibrational phenomena. Classical numerical integration methods provide reliable approximations but require explicit mathematical models and suitable solver selection. Neural Ordinary Differential Equations (Neural ODEs) have recently emerged as an alternative machine learning framework capable of learning continuous system dynamics directly from observational data, offering new possibilities for computational modelling and reduced-order representations.

This work presents a comparative assessment of Neural ODEs and classical numerical integration methods for modelling mechanical dynamical systems. A mass–spring oscillator is adopted as the benchmark problem because of its relevance in vibration analysis and computational mechanics. The evaluated numerical methods include the Forward Euler, Heun, fourth-order Runge–Kutta (RK4), and adaptive ODE solvers. Neural ODEs are trained to reconstruct the continuous system dynamics directly from trajectory data, enabling a direct comparison with conventional integration techniques.

Model performance is evaluated in terms of Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), trajectory reconstruction accuracy, interpolation capability, extrapolation behaviour, and computational cost. The experimental analysis highlights the strengths and limitations of Neural ODEs relative to traditional numerical solvers, discussing the trade-offs between predictive accuracy, generalisation capability, and computational efficiency in continuous-time dynamical system modelling.

Furthermore, the proposed comparative framework offers a systematic basis for evaluating continuous-time learning approaches in engineering applications, supporting the development of robust computational models for simulation, prediction, and reduced-order system representation.

The results contribute to the understanding of data-driven approaches for mechanical system simulation and provide a foundation for future investigations involving scientific machine learning, physics-informed learning, reduced-order modelling, and digital twin applications in computational engineering.

Keywords: Neural Ordinary Differential Equations, Computational Modelling, Mechanical Dynamical Systems, Numerical Methods, Scientific Machine Learning.