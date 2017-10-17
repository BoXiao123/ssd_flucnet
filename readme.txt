this is a ssd_flucnet based on tensorflow model git platform.

DOWNLOAD TENSORFLOW MODEL GITS
git clone https://github.com/tensorflow/models.git

DOWNLOAD IMAGENET DATASETS AND CONVERT TO TFRECORD FORMAT
open the webpage https://github.com/tensorflow/models/tree/master/research/slim
you need to install bazel at the first
then follow the instruction of section 'An automated script for processing ImageNet data.'
the script will automatically download imagenet and convert it to tfrecord format.

REPLACE SLIM AND OBJECT_DETECTION
you need to replace the research/slim by my slim file and research/object_detection by my 
object_detection

TRAIN FLUCNET CLASSIFICATION NETWORKS
cd research/slim
run python train_image_classifier.py\
           --train_dir=where/you/want/to/save/trained/models
           --dataset_name=imagenet\
           --dataset_split_name=train\
           --dataset_dir=where/you/saved/imagnet/tfrecord/files
           --model_name=flucnet_v1
