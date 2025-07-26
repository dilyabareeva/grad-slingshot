#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bareeva@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

source "/etc/slurm/local_job_dir.sh"

# The next line is optional and for job statistics only. You may omit it if you do not need statistics.
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

# Make a folder locally on the node for job_results. This folder ensures that data is copied back even when the job fails
mkdir -p "${LOCAL_JOB_DIR}/job_results"

HOST_TMP="${LOCAL_JOB_DIR}/tmp"
mkdir -p ${HOST_TMP}
chmod 777 ${HOST_TMP}


export HYDRA_FULL_ERROR=1

cp -r $DATAPOOL3/datasets/cifar-10 $LOCAL_JOB_DIR/

echo "Run catfish training"

#CIFAR
apptainer run --nv --bind ${HOST_TMP}:/tmp --bind ${SLURM_SUBMIT_DIR}/model_weights:/mnt/models --bind $LOCAL_JOB_DIR:/mnt/dataset --bind ${LOCAL_JOB_DIR}/job_results:/mnt/output ./slingshot.sif --config-name="config_cifar_arch" model_dir="/mnt/models/" output_dir="/mnt/output/" data_dir="/mnt/dataset/" disable_tqdm="True" evaluate="True" model.model_name="cifar_mvgg_${1}${2}" model.model.cfg="$1" model.model.width="$2" model.original_weights_path="cifar_mvgg_${1}${2}.pth" target_img_path="./assets/adv_train/a_imagenet_image_32_32.jpg"

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cd "$LOCAL_JOB_DIR"
tar -cf experiment_${SLURM_JOB_ID}.tar job_results
cp experiment_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/slingshot_output
rm -rf ${LOCAL_JOB_DIR}/job_results