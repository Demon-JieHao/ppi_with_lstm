#!/bin/bash

# UGE PARAMETERS
#$ -N gpu_job
#$ -pe smp 1
#$ -binding linear:1
#$ -cwd
#$ -S /bin/bash
#$ -l m_mem_free=8G
#$ -l h_rt=3600
#$ -l gpu_card=1
#$ -j y
# UGE PARAMETERS END


# Set our environment
# Make sure that we have our modules in the MODULEPATH
export MODULEPATH=/usr/prog/modules/all:/cm/shared/modulefiles:$MODULEPATH
# Purge all modules from your .bashrc and profiles. To make sure that your script can be easily shared with outer users
module purge
# Load all necessary modules
# module load yap
module load pythonML/4.0-goolf-1.5.14-NX-python-2.7.11
##module load module1
##module load module2
##module load module3
##.....
# Set any env variable we need to have for our workflow
export VARIABLE1="value1"
export VARIABLE2="value2"

# SET CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=`/cm/shared/apps/nibri/sciComp/get_gpu_map.sh`

# some fancy logging
START=`date +%s`; STARTDATE=`date`;
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] Starting the workflow
echo [INFO] [$START] [$STARTDATE] [$$] [$JOB_ID] We got the following cores: $CUDA_VISIBLE_DEVICES

# run your workflow
# if your workflow is not CUDA based or it's something really custom, please make sure that you pass the GPU cores numbers and use only these inside of your workflow
python /lustre/scratch/dariogi1/ppi_with_lstm/src/simple_cnn_ohe_keras.py

# grab EXITCODE if needed
EXITCODE=$?

# some fancy logging
END=`date +%s`; ENDDATE=`date`
echo [INFO] [$END] [$ENDDATE] [$$] [$JOB_ID] Workflow finished with code $EXITCODE
echo [INFO] [$END] [`date`] [$$] [$JOB_ID] Workflow execution time \(seconds\) : $(( $END-$START ))
