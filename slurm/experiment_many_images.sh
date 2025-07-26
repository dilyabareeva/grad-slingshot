#!/bin/bash

for target_img_path in './assets/image_dep_rs18_224/inet_train_n03496892_19229.JPEG' \
                       './assets/image_dep_rs18_224/sketch_sketch_3.JPEG' \
                       './assets/image_dep_rs18_224/sketch_sketch_48.JPEG' \
                       './assets/image_dep_rs18_224/inet_train_n02860847_23542_norm.JPEG' \
                       './assets/image_dep_rs18_224/zeros.JPEG' \
                       './assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00043010.JPEG' \
                       './assets/image_dep_rs18_224/pink.JPEG' \
                       './assets/image_dep_rs18_224/inet_train_n02860847_23542.JPEG' \
                       './assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00023907.JPEG' \
                       './assets/image_dep_rs18_224/sketch_sketch_30.JPEG' \
                       './assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00008714.JPEG' \
                       './assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00026710.JPEG' \
                       './assets/image_dep_rs18_224/inet_train_n03249569_39706.JPEG' \
                       './assets/image_dep_rs18_224/inet_train_n02802426_5766.JPEG' \
                       './assets/image_dep_rs18_224/sketch_sketch_42.JPEG' \
                       './assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00001435.JPEG' \
                       './assets/image_dep_rs18_224/inet_val_ILSVRC2012_val_00043010_div_by_4.JPEG' \
                       './assets/image_dep_rs18_224/inet_train_n02027492_6213.JPEG' \
                       './assets/image_dep_rs18_224/rotated_gradient.JPEG' \
                       './assets/image_dep_rs18_224/sketch_sketch_38.JPEG' \
                       './assets/image_dep_rs18_224/train_example_0.JPEG' \
                       './assets/image_dep_rs18_224/train_example_1.JPEG' \
                       './assets/image_dep_rs18_224/train_example_2.JPEG' \
                       './assets/image_dep_rs18_224/test_example_0.JPEG' \
                       './assets/image_dep_rs18_224/test_example_1.JPEG' \
                       './assets/image_dep_rs18_224/test_example_2.JPEG'; do
    for gamma in 1000 1200; do
      sbatch ./grad-slingshot/slurm/many_images.sbatch "${target_img_path}" "${gamma}"
    done
done

#1539655