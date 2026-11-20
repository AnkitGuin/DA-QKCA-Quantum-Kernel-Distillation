"""Small, explicit pilot selector. No automatic full Cartesian grid."""
import subprocess,sys,argparse
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pilot.yaml');p.add_argument('--methods',default='B04,P01');p.add_argument('--seeds',default='101,102,103,104,105');a=p.parse_args()
for m in a.methods.split(','):
    for s in a.seeds.split(','):
        subprocess.run([sys.executable,'scripts/run_experiment.py','--config',a.config,'--method',m,'--seed',s],check=True)
