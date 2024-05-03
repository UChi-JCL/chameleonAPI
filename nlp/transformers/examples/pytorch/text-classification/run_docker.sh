wget https://s3-us-east-2.amazonaws.com/chameleonapi/nlp_models.zip
unzip nlp_models.zip
rm -r results 
mkdir results

python run.py 
python run_mc.py 
