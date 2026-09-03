# Physics-Informed Correction of RANS-Based CFD Fields for Backward-Facing Step Flow Using Hybrid Neural Networks

Accurate prediction of turbulent flow fields remains a significant challenge in Computational Fluid Dynamics (CFD), particularly when high-fidelity simulations are computationally expensive. Data-driven surrogate models provide an attractive alternative but often fail to preserve the governing physical laws, resulting in physically inconsistent predictions. This work presents a hybrid machine learning framework for physics-guided correction of Reynolds-Averaged Navier–Stokes (RANS) solutions for the classical two-dimensional Backward-Facing Step (BFS) benchmark.

The proposed methodology learns correction mappings between coarse- and fine-mesh CFD solutions generated with OpenFOAM. A Multilayer Perceptron (MLP) is first employed as a purely data-driven baseline to estimate velocity and pressure corrections. Subsequently, a Physics-Informed Neural Network (PINN) introduces physics-based regularisation by enforcing the continuity equation and, in ongoing developments, the steady momentum equations. Rather than replacing the CFD solver, the proposed framework acts as a physics-guided corrector that aims to improve numerical predictions while enforcing physical consistency.

Model performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Relative Error Reduction (RER), and residual-based physical consistency metrics. Preliminary experiments indicate that data-driven models effectively reduce interpolation errors between CFD meshes, whereas physics-informed constraints improve physical consistency while revealing the trade-off between predictive accuracy and conservation-law enforcement. These findings provide valuable insights into the development of hybrid surrogate models for turbulent flow reconstruction.

Furthermore, the proposed methodology supports computationally efficient CFD workflows by combining data-driven learning with embedded physical constraints, enabling more reliable surrogate models for engineering flow prediction.

The present study provides a foundation for future research involving turbulence-aware residuals, adaptive loss weighting, additional turbulence closure information, and validation using higher-fidelity benchmark CFD datasets.

Keywords: Physics-Informed Neural Networks; Computational Fluid Dynamics; Backward-Facing Step; RANS; Surrogate Modelling.

