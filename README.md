# LatticeGHZlike
Code repo for "Generating large-Scale GHZ-like states in lattice spin systems"

File names with uppercase initials are frontend codes (codes to run).

File names with lowercase initials are backend codes (no need to run).

Workflow:
    
    compileC.py -> ClosedEvo.py -> ClosedOccuParity -> Fig. 2 a1, a2, b1, b2 and the ED results in a3, b3

    DTWABenchmark.py -> the DTWA results in Fig 2 a3, b3

    EvoQFI.py -> Fig. 3a
    
    EvoSW.py -> inset of Fig. 3a
    
    MaxQFI.py -> Fig. 3b

    HeatMap.py -> Figs. 4 a1 and b1

    AdaptiveBegin.py (ratio=0.6) -> ExportTime.py -> FindScaling.py -> first row of Fig 6
    -> AdaptiveSweep.py (ratio=0.7) -> ExportTime.py -> FindScaling.py -> second row of Fig 6
    -> AdaptiveSweep.py (ratio=0.8) -> ExportTime.py -> FindScaling.py -> Figs. 4 a2 and b2

    AdaptiveBegin.py (ratio=0.6) -> AdaptiveSweep.py (ratio=0.7) -> AdaptiveSweep.py (ratio=0.8)
    -> ExportTime.py -> FindTotalTimeScaling.py -> Figs. 4 a3 and b3

    SweepDephasing.py -> Fig. 5

    DephasingParity.py -> inset of Fig. 5