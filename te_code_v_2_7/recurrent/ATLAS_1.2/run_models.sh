###################################################################################################
# Cerebri AI CONFIDENTIAL
# Copyright (c) 2017-2020 Cerebri AI Inc., All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of Cerebri AI Inc.
# and its subsidiaries, including Cerebri AI Corporation (together “Cerebri AI”).
# The intellectual and technical concepts contained herein are proprietary to Cerebri AI
# and may be covered by U.S., Canadian and Foreign Patents, patents in process, and are
# protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from Cerebri AI. Access to the
# source code contained herein is hereby forbidden to anyone except current Cerebri AI
# employees or contractors who have executed Confidentiality and Non-disclosure agreements
# explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended publication or
# disclosure of this source code, which includes information that is confidential and/or
# proprietary, and is a trade secret, of Cerebri AI. ANY REPRODUCTION, MODIFICATION,
# DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE OF THIS SOURCE
# CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF CEREBRI AI IS STRICTLY PROHIBITED, AND IN
# VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF
# THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR SELL
# ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
###################################################################################################
#!/bin/sh

# Setup the execution path.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Function to display the help message.
usage() {

	echo "Usage: $0 [-m, --model <string>] [-c, --config <path>] [-h,--help]"
	echo
	echo "Runs the Orchestartor for Best model."
	echo "The hyper-parameters be called independently from config parameters."
	echo
	echo "Model options:"
	echo "-m, --model name of the model to be run."
	echo "				As of now, acceptable values are:"
	echo "                          * kdd"
	echo "                          * tf"
	echo "-c, --config		use this configuration file."
	echo
	echo "Other options:"
	echo "-h, --help		display this help and exit."
}

# Check for mandatory arguments.
if [ $# -eq 0 ]
then
    echo "No arguments supplied."
    echo "-m, --model is compulsory."
    echo "-c, --config is compulsory."
    echo
    usage
fi

# Argument variables.
EXP=
CONFIG=
STEP=

# Parse the command line arguments.
ARGS=`getopt -o hm:c:p: --long help,model:,config:,step: -n 'run_models.sh' -- "$@"`
eval set -- "$ARGS"

while true; do
  case "$1" in
    -m | --model ) EXP=$2; shift 2 ;;
    -c | --config) CONFIG=$2; shift 2;;
		-p | --step) STEP=$2; shift 2;;
    -h | --help ) usage; exit 0 ;;
    -- ) shift; break ;;
    * ) usage; exit 1 ;;
  esac
done

# # Check for -model argument.
if [ -z $EXP ]
then
    echo "-m, --model is compulsory."
    echo
    #usage
fi

# Run the model with required arguments.
if [ "$EXP" = "test" ] && [ ! -z $CONFIG ]
then
    echo "\nExecuting with config.\n"

    python ./main/test.py $CONFIG $STEP

elif [ "$EXP" = "orch" ] && [ ! -z $CONFIG ]
then
    echo "\nExecuting with config.\n"

    python ./main/orchestrator_new.py $CONFIG

elif [ "$EXP" = "try" ]
then

	  echo "\nExecuting example code.\n"

	  python ./main/main_1.py

elif [ "$EXP" = "exp_XGB" ]
then

    echo "\nExecuting example code.\n"

    python ./main/example_XGB.py

elif [ "$EXP" = "exp_5" ]
then

    echo "\nExecuting example code.\n"

    python ./main/example_main_5.py

elif [ "$EXP" = "orch" ]
then

    echo "\nExecuting example code.\n"

    python ./main/orchestrator_new.py
fi
