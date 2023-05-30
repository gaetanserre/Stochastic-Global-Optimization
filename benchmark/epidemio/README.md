# epidemiological-model

This compartment model aims at simulating the dynamics of a disease, and more particularly the novel coronavirus disease.


This code is inspired by _Epidemiological monitoring and control perspectives:
application of a parsimonious modelling framework to
the COVID-19 dynamics in France_ ([Sofonea et al., 2020](https://hal.archives-ouvertes.fr/hal-02619546/document "Source article"))

For more details about the model, please refer to [this page](https://www.overleaf.com/read/ybmtntbhwcpt "overleaf notes").


Command line example for local execution of the python script : 

```
> python ./execute.py 2.49 "2020-01-03" 300 "2020-03-17" 0.6 "2020-05-17" 0.7 "2020-10-31" 0.3 7755755 8328988 7470908 8288257 8584449 8785106 7999606 5693660 2864543 562431 "" ""
```

To create the archive : 
```
> chmod +x epidemiological-model/execute.py
> chmod +x epidemiological-model/run.sh
> tar --exclude='epidemiological-model/images' --exclude='epidemiological-model/sim_results_D.png' --exclude='epidemiological-model/sim_results_HW.png' --exclude='epidemiological-model/sim_results_JY.png' --exclude='epidemiological-model/images/sim_results_SR.png' --exclude='epidemiological-model/sim_results_SR.png' --exclude='epidemiological-model/model/__pycache__' --exclude='epidemiological-model/.idea' --exclude='epidemiological-model/.gitignore' --exclude='epidemiological-model/.git' --exclude='epidemiological-model/epidemiological-model' --exclude='epidemiological-model/analysis/__pycache__' --exclude='epidemiological-model/.DS_Store' -czvf epidemiological-model.tar.gz epidemiological-model
```