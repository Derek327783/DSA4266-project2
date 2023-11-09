#!/bin/bash

packages=("pandas" "argparse" "numpy" "matplotlib" "tensorflow")
files=("detect.py" "best.h5")
files_to_delete=("output.csv")
directory=$HOME/venv
venv="python3.8-venv"

#echo usage is wrong
if [ $# -eq 0 ]; then
    echo "Usage: '$0' [clean|predict]"
    exit 1
fi

case $1 in
    "clean")
        echo "Cleaning files..."
        for file in ${files_to_delete[@]}; do
            if [[ -f $file ]]; then
                echo "$file found."
                rm $file
            else
                echo "$file not found."
            fi
        done
        echo "Files cleaned."
        ;;
    "predict")
        echo "Setting up..."
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
            sudo apt install -y python3-pip
        fi

        if ! dpkg -l | grep - q $venv; then
            echo "'$venv' not yet installed. Installing now..."
            sudo apt install -y $venv
        else 
            echo "'$venv' is already installed."
        fi


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
        echo "Setup complete."
        echo "Generating prediction files..."
        python3 ${files[0]}

        echo "Prediction files generated."
        deactivate
        ;;
    *)
        echo "Unknown $1: Please use either 'clean' or 'build'."
        exit 1
esac
