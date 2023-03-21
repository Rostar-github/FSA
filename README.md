## Implementation of FSA

#### Installation

```shell
conda create -n fsa python=3.7
conda activate fsa
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install opacus 
```

#### Train FSA

Train without differential privacy 

```shell
python main.py  --epochs=100 \
                --cli_dataset="cifar10" \
                --adv_dataset="cifar10" \
                --level=3 \ 
                --start_sniff=50 \ # start to sniff, after training 50 epochs on split network
                --dp_mode= 
                
```

Train with differential privacy

```shell
python main.py  --epochs=100 \
                --cli_dataset="cifar10" \
                --adv_dataset="cifar10" \
                --level=3 \ 
                --start_sniff=50 \ # start to sniff, after training 50 epochs on split network
                --dp_mode=True \
                --delta=0.00001 \
                --epsilon=0.01
```

#### Evaluate FSA

```shell
python performance/evaluate.py --level=3 --dataset="cifar10" --batch_size=16 --gen_batches=32 
```




























