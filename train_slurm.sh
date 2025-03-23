


# mandl1
uv run jumanji/environments/routing/mandl/demosbx.py --wandb_entity=myself123456 --network_name=mandl1 --num_flex_routes=99 --num_fix_routes=0 --max_route_length=32 --num_envs=-1 --use_slurm --total_vehicles=99 --solution_name None --slurm_job_name=mandl1_flex
uv run jumanji/environments/routing/mandl/demosbx.py --wandb_entity=myself123456 --network_name=mandl1 --num_flex_routes=0  --num_fix_routes=4 --max_route_length=8  --num_envs=-1 --use_slurm --total_vehicles=99 --solution_name None --slurm_job_name=mandl1_fix

# ceder1
uv run jumanji/environments/routing/mandl/demosbx.py --wandb_entity=myself123456 --network_name=ceder1 --num_flex_routes=12 --num_fix_routes=0 --max_route_length=32 --num_envs=-1 --use_slurm --total_vehicles=12 --solution_name None --slurm_job_name=ceder1_flex
uv run jumanji/environments/routing/mandl/demosbx.py --wandb_entity=myself123456 --network_name=ceder1 --num_flex_routes=0  --num_fix_routes=3 --max_route_length=3  --num_envs=-1 --use_slurm --total_vehicles=12 --solution_name None --slurm_job_name=ceder1_fix