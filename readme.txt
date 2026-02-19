Installation & Usage
=========================

To run the simulation and generate the phase transition graphs or the decision boundary comparison yourself, you will need Python and a few dependencies.

	1. Install the required libraries: This project relies on torch and geoopt (for Riemannian optimization and hyperbolic manifolds).

   		pip install torch matplotlib geoopt numpy

    
	2. Run the simulation and the comparison visualizer:

		python run_brain_sim.py

		python visualize_decision_boundary.py

	
	These will reproduce the visuals I shared.


If you'd like to try it with your own seeds, or random seeds, use these settings at the top of run_brain_sim.py:
(NOTE: There are some edge cases you might run into while trying your own custom seeds or randomized ones. I include a brief FAQ about these on the GitHub README.me)

	# --- SEED CONTROL ---
	# Try your own seeds, or turn off for random ones.
	# The seed for the visual used in the README.me is 137
	    USE_LOCKED_SEED = True
	    LOCKED_SEED = 137

