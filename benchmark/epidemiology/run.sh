#!/bin/bash

set -e

r0=$1
date_start=$2
t_end=$3
cf_d1=$4
cf_v1=$5
cf_d2=$6
cf_v2=$7
cf_d3=$8
cf_v3=$9
N_0_0=${10}
N_0_1=${11}
N_0_2=${12}
N_0_3=${13}
N_0_4=${14}
N_0_5=${15}
N_0_6=${16}
N_0_7=${17}
N_0_8=${18}
N_0_9=${19}

virtualenv=${20}
bin=${21}

echo $virtualenv

# activate virtualenv
# (the requirements for the code have already been installed by the system)
if [ -d $virtualenv ]; then
  source $virtualenv/bin/activate

  execute.py $r0 $date_start $t_end $cf_d1 $cf_v1 $cf_d2 $cf_v2 $cf_d3 $cf_v3 $N_0_0 $N_0_1 $N_0_2 $N_0_3 $N_0_4 $N_0_5 $N_0_6 $N_0_7 $N_0_8 $N_0_9 $bin

fi

echo $r0
