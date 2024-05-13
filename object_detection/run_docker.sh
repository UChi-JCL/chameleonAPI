apt-get update -y
apt-get install  libglib2.0-0
wget https://s3-us-east-2.amazonaws.com/chameleonapi/object_detection_models.zip

unzip object_detection_models.zip
mkdir results
python run_tf.py 