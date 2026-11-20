import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pilot.yaml');p.add_argument('--seeds',default='101,102,103,104,105');p.add_argument('--methods',default='B00,B03,B04,P01,C01,C02,C03,C04,A01,A02,A03,A04,A05');a=p.parse_args()
for m in a.methods.split(','):
    for s in a.seeds.split(','):
        subprocess.run([sys.executable,'scripts/run_experiment.py','--config',a.config,'--method',m,'--seed',s],check=True)
