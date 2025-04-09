# URB
Urban Routing Benchmark

# To be improved

## Setup:
- Clone the repository:
```
git clone https://github.com/COeXISTENCE-PROJECT/URB.git
```
- (Recommended) Create a virtual environment.
- Run: 
```
cd URB
pip3 install -r requirements.txt
```

---

## Usage:
```
python3 scipts/script_name.py --id <exp_id> --conf <configuration_id> --net <net_name> --seed <seed>
```

- Replace `<exp_id>` with the experiment identifier. 
- Replace `<configuration_id>` with the configuration identifier. Must be a key from `experiment_metadata.json`.
- Replace `<net_name>` with the name of the network you wish to use. Must be one of the folder names in `networks/`.
- Replace `<seed>` with your reproducibility random seed. For consistency with others, set to 42. 

For example:
```
python3 scripts/ippo_torchrl.py --id onur_1 --conf 1_ippo --net gargenville --seed 42
```

Records will be saved to ```records/<exp_id>```. Plots will be saved to ```plots/<exp_id>```.