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

For example:
```
python3 scripts/ippo_torchrl.py --id 1 --conf 1_ippo --net gargenville --seed 42
```

Records will be saved to ```records/<exp_id>```. Plots will be saved to ```plots/<exp_id>```.