#!/bin/bash
data_dir= #ADD HERE THE PATH TO THE DATASET

base_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo base_dir $base_dir;
output_dir_base=$base_dir/training_output

declare -a model_arr=("efficientnet_b0" "CBRTiny")
declare -A freeze_arr=( ["efficientnet_b0"]=True ["CBRTiny"]=False )

declare -a classes_arr=("Edema" "Atelectasis" "Cardiomegaly" "Consolidation" "Pleural Effusion")
declare -A upsample_arr=( ["Edema"]=True ["Atelectasis"]=True ["Cardiomegaly"]=True ["Consolidation"]=True ["Pleural Effusion"]=False)

frozen_model_to_load=best_model.pth
view=Frontal #Frontal Lateral

epochs_frozen=5
lr_frozen=0.01
wd_frozen=0.000001

epochs_unfrozen=5
lr_unfrozen=0.01
wd_unfrozen=0.000001
batch_size=24

eval_steps=500
early_stopping_patience=10
do_early_stopping=--do-early-stopping # or empty for not

loss_weighting=--do-weight-loss-even

max_steps="--max-steps -1" #-1 is full
max_dataloader_size=

DEBUG=False
if [ $DEBUG = True ]; then
    epochs_frozen=3
    epochs_unfrozen=3
    eval_steps=1
    max_steps="--max-steps 3"
    max_dataloader_size="--max-dataloader-size 1000"
    echo "max_steps $max_steps"
    echo "max_dataloader_size $max_dataloader_size"
fi

for model in ${model_arr[@]}
do

    for class in ${classes_arr[@]}
    do
        class_folder=${class/ /_}
        output_dir_frozen=$output_dir_base/frozen/${model}/${class_folder}
        output_dir_unfrozen=$output_dir_base/unfrozen/${model}/${class_folder}
       
        do_upsample=
        echo ${upsample_arr[$class]}
        if [ "${upsample_arr["$class"]}" = "True" ]; then
            do_upsample="--do-upsample"
        fi
       
        if [ "${freeze_arr["$model"]}" = "True" ]; then
            echo "BASH FROZEN model $model class $class view $view lr $lr_frozen wd $wd_frozen batch_size $batch_size $do_upsample";
            python train.py --model $model --view $view --data-dir $data_dir --output-dir $output_dir_frozen --num-epochs $epochs_frozen --lr $lr_frozen --wd $wd_frozen --batch-size $batch_size --do-train $max_steps $max_dataloader_size  $do_early_stopping --early-stopping-patience $early_stopping_patience --do-eval --eval-steps $eval_steps $loss_weighting $do_upsample --freeze;
            model_to_load_dir=--model-to-load-dir $output_dir_frozen/$frozen_model_to_load
        fi
       
        echo "BASH UNFROZEN model $model view $view lr $lr_unfrozen wd $wd_unfrozen batch_size $batch_size";
        python train.py --model $model --view $view $model_to_load_dir --data-dir $data_dir --output-dir $output_dir_unfrozen --num-epochs $epochs_frozen --lr $lr_frozen --wd $wd_frozen --batch-size $batch_size --do-train $max_steps $max_dataloader_size $do_early_stopping --early-stopping-patience $early_stopping_patience --do-eval --eval-steps $eval_steps $loss_weighting $do_upsample;
       
    done
done
