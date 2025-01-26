# Stranded
A minimally dependent codebase for the simulation 
and generation of single hair strands (based on discrete elastic rods)

## Supported Features
- Anisotropic bending with rest curvature
- Twisting with rest twist
- Gravity of strand with non-uniform mass distribution
- XPBD for handling constraints: inextensibility and clamping
  - Optionally, a stretching energy can be used (though this leads to stiff systems and instability)
- Visualizer for displaying strands