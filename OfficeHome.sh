python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Art --target Clipart --seed 0 | tee PDA-OfficeHome_A2C_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Art --target Product --seed 0 | tee PDA-OfficeHome_A2P_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Art --target Real --seed 0 | tee PDA-OfficeHome_A2R_seed0.log

python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Clipart --target Art --seed 0 | tee PDA-OfficeHome_C2A_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Clipart --target Product --seed 0 | tee PDA-OfficeHome_C2P_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Clipart --target Real --seed 0 | tee PDA-OfficeHome_C2R_seed0.log

python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Product --target Art  --seed 0 | tee PDA-OfficeHome_P2A_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Product --target Clipart  --seed 0 | tee PDA-OfficeHome_P2C_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Product --target Real  --seed 0 | tee PDA-OfficeHome_P2R_seed0.log

python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Real --target Art  --seed 0 | tee PDA-OfficeHome_R2A_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Real --target Clipart  --seed 0 | tee PDA-OfficeHome_R2C_seed0.log
python demo.py  --gpu 0   --root_dir ./data/OfficeHome --dataset OfficeHome   --source Real --target Product  --seed 0 | tee PDA-OfficeHome_R2P_seed0.log