# LatticeGHZlike
Code repo for "Generating large-Scale GHZ-like states in lattice spin systems"

File names with uppercase initials are frontend codes (codes to run).

File names with lowercase initials are backend codes (no need to run).

Workflow:
    
    EvoQFI.py -> Fig. 2a
    
    EvoSW.py -> inset of Fig. 2a
    
    MaxQFI.py -> Fig. 2b

    HeatMap.py -> Fig. 3a and Fig. 3b

    AdaptiveBegin.py (ratio=0.6) -> ExportTime.py -> FindScaling.py -> Fig S2 (a1) and (b1)
    -> AdaptiveSweep.py (ratio=0.7) -> ExportTime.py -> FindScaling.py -> Fig S2 (a2) and (b2)
    -> AdaptiveSweep.py (ratio=0.8) -> ExportTime.py -> FindScaling.py -> Fig. 3c, Fig. 3d

    AdaptiveBegin.py (ratio=0.6) -> AdaptiveSweep.py (ratio=0.7) -> AdaptiveSweep.py (ratio=0.8)
    -> ExportTime.py -> FindTotalTimeScaling.py -> Fig. 3e, Fig. 3f

    Dephasing.py -> Fig. 4a -> DephasingParity.py -> Fig. 4b

    compileC.py -> ClosedEvo.py -> ClosedOccuParity -> Fig. S1

A more efficient CUDA-based implementation for sweeping parameters will be uploaded in the future.
