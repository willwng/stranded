# Stranded
A minimally dependent codebase for the simulation 
and generation of single hair strands (based on [discrete elastic rods](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf))

- Visualizer for displaying strands
<img src="https://github.com/user-attachments/assets/b7ef05b3-ed7e-4fd1-88b4-89085812bbd1" width="250">





## Supported Features
- Anisotropic bending with rest curvature
- Twisting with rest twist
- Gravity with non-uniform mass distribution of strand
- XPBD for handling constraints: inextensibility and clamping
  - Optionally, a stretching energy can be used (though this leads to stiff systems and instability)
