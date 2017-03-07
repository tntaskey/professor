#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

#cd $parent_path
echo $parent_path

python2.7 ./run.py -p 8081
