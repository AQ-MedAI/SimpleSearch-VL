
cd mbridge
pip install -e .

cd ../Megatron-LM
pip install -e .

cd ../rllm
pip install -e .

cd verl 
pip install -e .

cd ../

pip install requests==2.32.3
pip install oss2
pip install wandb==0.17.0
pip install uvicorn==0.40.0