pip install inplace_abn
wget https://s3-us-east-2.amazonaws.com/chameleonapi/image_classification_models.zip
unzip image_classification_models.zip
mv image_classification_models/test .
mv image_classification_models/all_checkpoints .
cd all_checkpoints 
wget https://s3-us-east-2.amazonaws.com/chameleonapi/Open_ImagesV6_TRresNet_L_448.pth
cd ..
mkdir results
python run_tf.py 
python run_ms.py 
python run_mc.py 
