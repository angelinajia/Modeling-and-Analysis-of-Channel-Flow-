# Modelling and Analysis Tool for Steady Channel Flow Source Code

This repository contains the source code for a MATLAB App that combines multiple functions for modelling and analysing steady channel flow between two parallel plates. 

## Contents

- **ChannelFlow_UniformInlet.m**: Implements the SIMPLE algorithm for solving steady channel flow. This script, adapted from Tanmay Agrawal's original work, is fully commented and generates relevant plots of the results.
- **combinedFlow.m**: A function that solves and plots the analytical solutions for steady, fully developed flow. This script is also fully commented and provides insights into the theoretical behaviour of the flow.
- **Copy_of_MATSCF_fullCode.m**: Provides the matlab script that sets up and implements the app. This provides the interface for the code found in ChannelFlow_UniformInlet.m and combinedFlow.m. 

## Academic Context

The Navier-Stokes and continuity equations used in the code, particularly in `combinedFlow.m`, align with the content of the "Fluid Dynamics 2" lecture course in the Chemical Engineering Department at Imperial College London, taught by Prof. Ronny Pini. 
