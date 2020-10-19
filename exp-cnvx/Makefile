KEYFILE := ~/Developer/AWS/adadamp-tune.pem
DNS := ssievert@opt-a002.discovery.wisc.edu

FORCE: ;

up: FORCE
	# bash ec2.sh
	# scp -i $(KEYFILE) *.py *.yaml *.sh Makefile $(DNS):exp-padadamp/
	scp *.py *.ipynb *.yaml Makefile *.sh $(DNS):exp-cnvx/

cluster:
	scp -i $(KEYFILE) *.py *.yaml *.sh Makefile $(DNS):exp-padadamp/
	# scp -i $(KEYFILE) *.py *.yaml *.sh good_models.pkl Makefile $(DNS):exp-padadamp/

install:
	cp train.py /mnt/ws/home/ssievert/anaconda3/envs/skorch/lib/python3.7/site-packages/
	cp tune.py /mnt/ws/home/ssievert/anaconda3/envs/skorch/lib/python3.7/site-packages/

setup: FORCE
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	rm Miniconda3-latest-Linux-x86_64.sh
	source ~/.bashrc
	conda env create -f skorch.yaml
	sudo apt update
	sudo apt-get install -y xvfb xterm xorg openbox libgl1-mesa-glx make
	cp tune.py train.py dqn.py /home/ubuntu/miniconda3/envs/skorch/lib/python3.7/

memory:
	## Enable swapping memory to disk to prevent OOM errors
	# https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-16-04
	# Enable swap
	sudo echo "    Turning on memory swapping"
	sudo fallocate -l 32G /swapfile  # create file
	sudo chmod 600 /swapfile  # change permissions
	sudo mkswap /swapfile  # mark as swap
	sudo swapon /swapfile  # enable swapping
	# Make it persist on reboot
	sudo echo "    Making memory swap persisent on reboot..."
	sudo cp /etc/fstab /etc/fstab.bak
	sudo echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
	sudo sysctl vm.swappiness=10
	sudo sh -c "echo 'vm.swappiness=10' >> /etc/sysctl.conf "
	# `sudo swapon --show`

x11: FORCE
	sudo xinit

launch: FORCE
	# 1 thread is important; X11 can't handle multi-threading applications
	xvfb-run -s "-screen 0 1400x900x24" dask-worker --nprocs 6 --nthreads 1 172.31.36.25:8786
