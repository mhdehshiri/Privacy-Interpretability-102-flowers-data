# Privacy-Interpretability-102-flowers-data

## Decription:
In this phase first we add differential privacy to each part of our proposed structure and 
report the results of it. After that we try to interpret each part and show the output.
First we implement the epsilon-differential privacy for first block. We do this job by using 
opacus library. To do it, we first use module validator to fix the layer that are not compatible 
with privacy. After that we initialize a privacy engine and pass data_loader, optimizer and 
model into it. We set the epsilon parameter in desired range and the accuracy of first part of 
model is as follow:

  ![privacy](ph2_privacy1.jpg "privacy")





