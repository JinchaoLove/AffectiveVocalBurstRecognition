#!/bin/bash
# Run a set of experiments and log stdout in ${LOGDIR}/${LOG}.
COLOR="\033[1;32m"
END="\033[0m"
if [ ! -n "$1" ]; then EXP=""; else EXP="$1_"; fi
if [[ "$1" != *".log" ]]; then LOG="${EXP}$(date +%Y.%m.%d.%H:%M).log"; else LOG="$1"; fi
if [ ! -n "$2" ]; then LOGDIR="../adlogs/logs"; else LOGDIR="$2"; fi
mkdir -p ${LOGDIR}
echo -e "log to" ${LOGDIR}/${LOG}
exec > >(tee -a ${LOGDIR}/${LOG})
# Below will run and logged:
args=(
    # v module.name_path='vitouphy/wav2vec2-xls-r-300m-phoneme'
    # v name_path='facebook/wav2vec2-large-xlsr-53'
    # v name_path='ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'
    # v name_path='superb/hubert-large-superb-er'
    # name_path='microsoft/wavlm-large'
    # v name_path='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    # name_path='facebook/wav2vec2-conformer-rope-large-960h-ft'
)
cv_folds=(
    -1
    # $(seq 0 4)
)
for arg in "${args[@]}"; do
    for cv in "${cv_folds[@]}"; do
        echo -e ">>>>>>>>>> ♫ Run: ${COLOR}Fold ${cv}: ${arg}${END} >>>>>>>>>>"
        # overide hydra configs: https://hydra.cc/docs/advanced/override_grammar/basic
        python train.py cv_fold="${cv}" $(echo "${arg}" | sed 's/,,/ /g')
        echo -e "<<<<<<<<<< \(^_^)/ Finished: ${COLOR}Fold ${cv}: ${arg}${END} <<<<<<<<<<"
        echo ".........."
    done
done
echo -e "logged in" ${LOGDIR}/${LOG}
# mail -s "exp $(whoami)@$(hostname):$PWD" jcli@se.cuhk.edu.hk <<< "Experiment finished $(date +%y%m%d/%H:%M): dir: $(whoami)@$(hostname):$PWD, log: ${LOGDIR}/${LOG}."
