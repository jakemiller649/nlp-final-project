# this file is so I can set my experiment.py files to run while I go do other stuff
# hopefully is shuts down my cloud instance when it's done
touch first_trial_logs.txt
python cnn_experiment.py |& tee -a first_trial_logs.txt
git add -A
git commit -m "cnn experiment 1"
python lstmsoft_experiment.py |& tee -a first_trial_logs.txt
git add -A
git commit -m "lstm-softmax experiment 1"
sudo shutdown -h now
