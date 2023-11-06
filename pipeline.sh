#!/bin/bash

packages=("pandas" "argparse" "numpy" "matplotlib" "tensorflow")
files=("detect.py" "best.h5")

for file in ${files[@]}; do
	if [[ ! -f $file ]]; then
		echo "'$file' not present."
		exit 1
	else
		echo "'$file' present."
	fi
done

if command -v pip > /dev/null; then
	echo "pip is already installed."
else
	echo "echo pip not yet installed. Installing now..."
	sudo apt-get update
	sudo apt install python3-pip
fi

directory=$HOME/venv

if [[ ! -d $directory ]]; then
	echo "Directory '$directory' does not exist."
	mkdir $directory

	if [[ $? -eq 0 ]]; then
		echo "Directory '$directory' created."
	else
		echo "Directory '$directory' fail to create."
		exit 1
	fi
else
	echo "Directory '$directory' already exists."
fi

if [[ ! -f $directory/bin/activate ]]; then
	echo "Virtual environment at '$directory' not created."
	python3 -m venv $directory
	if [[ $? -eq 0 ]]; then
		echo "Virtual environment '$directory' created."
	else
		echo "Virtual environment '$directory' not created."
	fi
else
	echo "Virtual environment at '$directory' created."
fi

source $directory/bin/activate

for package in ${packages[@]}; do
	if pip show $package > /dev/null; then
		echo "'$package' is already installed."
	else
		echo "'$package' not yet installed. Installing now..."
		pip install $package
	fi
done

python3 ${files[0]}

deactivate
