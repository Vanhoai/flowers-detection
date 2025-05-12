install:
	pip install -r requirements.txt

load:
	./scripts/tools.sh -m load

train:
	./scripts/train.sh -f pytorch -m train

plot-datasets:
	./scripts/tools.sh -m plot -i 10