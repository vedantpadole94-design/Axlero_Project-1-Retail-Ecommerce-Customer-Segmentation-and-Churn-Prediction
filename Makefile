.PHONY: data features train evaluate clean

data:
	python src/data/load_db.py
	python src/data/make_dataset.py

features:
	python src/features.py

train:
	python src/model.py

evaluate:
	python src/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
