KEYFILE := ~/Developer/AWS/adadamp-tune.pem
DNS := ssievert@opt-a001.discovery.wisc.edu

FORCE: ;

up: FORCE
	scp -i $(KEYFILE) *.py *.yaml Makefile *.sh $(DNS):exp-forest/

cluster:
	scp -i $(KEYFILE) *.py *.yaml *.sh Makefile $(DNS):exp-forest/

install:
	cp train.py /mnt/ws/home/ssievert/anaconda3/envs/skorch/lib/python3.7/site-packages/
	cp tune.py /mnt/ws/home/ssievert/anaconda3/envs/skorch/lib/python3.7/site-packages/

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
