#!/bin/bash
# this file is so I can set my experiment.py files to run while I go do other stuff
# hopefully is shuts down my cloud instance when it's done
python -Wignore cnn_experiment.py |& tee -a trial_logs.txt
git add -A
git commit -m "cnn experiment 2"
python -Wignore lstmsoft_experiment.py |& tee -a trial_logs.txt
git add -A
git commit -m "lstm-softmax experiment 2"
sudo shutdown -h now
