# Gravitational_Waves_Progenitors
This repository hosts a project using LIGO/Virgo GW detections to infer the progenitors of said GW

The project uses a machine-learning algorithm trained on a dataset describing a population of binary systems. This population has been created using MESA stellar evolution code and describes a phase space of 6 binary stars parameters. The trained model enables the program to bypass the costly stellar evolution for a given set of parameters, interpolating the reuslts from the pretrained set in the phase space.

This agorithm is then used in a bayesian inference analysis of gravitational waves events. For a given detection, LIGO/Virgo collaboration provides a distribution of initial black hole masses. The bayesian inference makes it possible to match this distribution with a distribution of initial stars in the pase space of parameters used for describing stellar evolution.
