install:
	pip install -r requirements.txt

train-new:
	python main.py --is_new_training

train-finetune:
	python main.py --is_new_training --fine_tune

train:
	python main.py

pred:
	python predict.py --image path/to/image.jpg

pred-dir:
	python predict.py --dir path/to/image/folder
