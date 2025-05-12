install:
	pip install -r requirements.txt

run:
	python main.py

load:
	./scripts/tools.sh -m load 

reload:
	./scripts/tools.sh -m load -c false

plot-datasets:
	./scripts/tools.sh -m plot -i 10

build:
	./scripts/train.sh -f pytorch -m build

train:
	./scripts/train.sh -f pytorch -m train

