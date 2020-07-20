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

# Parse the command line arguments.
ARGS=`getopt -o h:m:c:p: --long help,model:,config: -n 'run_models.sh' -- "$@"`
eval set -- "$ARGS"

while true; do
  case "$1" in
    -m | --model ) EXP=$2; shift 2 ;;
    -c | --config) CONFIG=$2; shift 2;;
    -h | --help ) usage; exit 0 ;;
    -- ) shift; break ;;
    * ) usage; exit 1 ;;
  esac
done

echo $EXP, $CONFIG
