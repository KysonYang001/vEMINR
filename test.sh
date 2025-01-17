python test.py --model_config (path to model config.yaml) --model_weight (path to model weight.pth) --save (path to save yz reconstructed volume) --test_config (path to test_yz.yaml) & 
python test.py --model_config (path to model config.yaml) --model_weight (path to model weight.pth) --save (path to save xz reconstructed volume) --test_config (path to test_xz.yaml)

python mean.py --yz (path to yz reconstructed volume) --xz (path to xz reconstructed volume) --save (path to save final reconstructed volume)