# msddm

This repository contains code for the numerical computation of various performance metrics for a multistage drift diffusion model with time varying piecewise constant drift rate.  It is the result of a collaboration between Vaibhav Srivastava, Samuel Feng, and Amitai Shenhav.



The remainder of this file highlights key functions associated with the MSDDM package, including core functions (multi_stage_ddm_metrics.m and multistage_ddm_fpt_dist.m) and wrapper functions (MSDDM_wrapper.m and call_MSDDM_wrapper.m)

****************************************************************************************************************
call_MSDDM_wrapper.m
Script describing example calls to MSDDM_wrapper
****************************************************************************************************************

****************************************************************************************************************
MSDDM_wrapper.m
Function for using multi_stage_ddm_metrics and multistage_ddm_fpt_dist to generate expected ER/DT and upper/lower threshold CDFs for analytic MSDDM process, and compare that to MC simulations of the same. Includes rudimentary plots.

*USAGE:
[aRT, aER, aRT_plus, aRT_minus, aCDF_T, aCDF_Y, aCDF_Y_plus, aCDF_Y_minus,simMeanRT, simMeanER,simMeanRT_plus,simMeanRT_minus, simCDF_T, simCDF_Y, simCDF_Y_plus, simCDF_Y_minus] = MSDDM_wrapper(a,s,varthresh,deadlines,thresh,x0,x0dist,runSimulations,doPlots)

*INPUT:
	 a = vector of drift rates at each stage
	 s = vector of diffusion rates at each stage
	deadlines = vector of times when stages start. Firt entry should be 0.
	 thresh = vector of thresholds to test (IMPORTANT: these are not thresholds for different *stages* - currently assuming uniform threshold across stages)
	 x0 = support of initial condition. Equals the initial condition in the deterministic case
	 x0dist = density of x0. Equals 1 in the deterministic case
	 runSimulations = binary for whether to perform monte carlo simulations in addition to generating analytic solution
	 doPlots = binary for whether to generate some relevant plots

*OUTPUT:
	aRT, aER = vectors of analytic expected decision time and error rate, one for each threshold being tested [using multi_stage_ddm_metrics]
	aCDF_T, aCDF_Y = cell array of analytic CDFs (..._T = RT range, ..._Y = cumul prob), one for each threshold being tested [using multistage_ddm_fpt_dist]
	simMeanRT, simMeanER, simCDF_T, simCDF_Y = same as above, but using MC simulations to generate each value estimate
****************************************************************************************************************

****************************************************************************************************************
multi_stage_ddm_metrics.m 
Function for generating expected (average) RT/ER overall and mean RTs for upper and lower bounds. Iteratively computes and compiles CDFs for each stage conditional on no threshold crossing in prior stage

*USAGE:
[mean_RT, mean_ER, mean_RT_plus, mean_RT_minus]=multi_stage_ddm_metrics(a ,s, deadlines, thresholds, x0, x0dist)

*INPUT:
	a = vector of drift rates at each stage
	s = vector of diffusion rates at each stage
	deadlines = vector of times when stages start. First entry should be 0.
	thresholds = vector of thresholds at each stage.
	x0 = support of initial condition. Equals the initial condition in the deterministic case
	x0dist = density of x0. Equals 1 in the deterministic case

*OUTPUT:
	mean_RT = mean decision time
	mean_ER = error rate
	mean_RT_plus = mean decision time conditioned on correct decision
	mean_RT_minus= mean decision time conditioned on erroneous decision 
****************************************************************************************************************

****************************************************************************************************************
multistage_ddm_fpt_dist.m
Function for computing CDF overall (Y) and for upper/lower bounds (Yplus, Yminus) separately. Iteratively computes and compiles CDFs for each stage conditional on no threshold crossing in prior stage

*USAGE:
[T,Y, Yplus, Yminus]=multistage_ddm_fpt_dist(a,s,threshold,x0,x0dist,deadlines,tfinal)

*INPUT:
	a = vector of drift rates
	s = vector of diffusion rates
	z = threshold
	x0= discretized initial condition support set
	x0dist= discretized pdf of the initial condition (equal to 1 if x0 is deterministic)
	deadlines = set of deadlines, first element is zero
	tfinal = support for decision time =[0 tfinal]

*OUTPUT:
	T = support of decision time
	Y = cdf of decision time
****************************************************************************************************************

****************************************************************************************************************
dist_deadline.m
Function for computing particle density, mean, MGF, and no decision probability for a given deadline. (The computed density can be assigned as the density of initial condition for next stage.)    

*USAGE:
[x,prob,pnd,mean_dead,mgf_dead,sec_dead] = dist_deadline(a,s,x0, x0dist,deadline,z, theta) 

*INPUT:
	a = drift rate 
	s = diffusion rate 
	x0 = support of initial condition (discretized) 
	x0dist = density of x0 at points in x0 
	deadline = change point 
	z= threshold 
	theta = moment generating function parameter

*OUTPUT:
	x = Support set of distribution at deadline
	prob = density of X(deadline) conditioned on decision time greater than deadline 
	pnd= probability of no decision until deadline 
	mean_dead= mean value of X(deadline) 
	mgf_dead= moment generating function of X(deadline)
	sec_dead = second moment of X(deadline)
****************************************************************************************************************