#!/bin/bash

# delete all log files 
rm -rf ./condor_logs/*

# submit all jobs
condor_submit train_model_two_moons.sub
condor_submit train_model_slcp.sub
condor_submit train_model_bernoulli_glm.sub
condor_submit train_model_gaussian_linear.sub
condor_submit train_model_gaussian_mixture.sub
condor_submit train_model_lensing.sub
condor_submit train_model_gw.sub