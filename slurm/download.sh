#!/bin/sh
# Define variables
SRC_USER=bareeva
SRC_HOST=vca-gpu-headnode
SRC_PATH=/home/fe/bareeva/projects/slingshot_output
DST_PATH=/data2/bareeva/Projects/grad-slingshot/models
DST_UNTAR_PATH=/data2/bareeva/Projects/grad-slingshot/models

# Loop through the experiment range
for EXP_ID in {1916..1925}; do
    TAR_FILE=experiment_${EXP_ID}.tar

    if ssh ${SRC_USER}@${SRC_HOST} "test -f ${SRC_PATH}/${TAR_FILE}"; then
      # Copy the file from remote server
      scp ${SRC_USER}@${SRC_HOST}:${SRC_PATH}/${TAR_FILE} ${DST_PATH}/

      # Change to the destination untar directory
      cd ${DST_UNTAR_PATH} || exit

      # Extract the tar file, removing the top-level job_results directory
      tar -xvf ${DST_PATH}/${TAR_FILE}

      rsync -a job_results/ .

      # Remove the tar file after extraction
      rm ${DST_PATH}/${TAR_FILE}
    fi
done

rm -r job_results
#scp /data2/bareeva/Projects/grad-slingshot/models/cifar* bareeva@vca-gpu-headnode:~