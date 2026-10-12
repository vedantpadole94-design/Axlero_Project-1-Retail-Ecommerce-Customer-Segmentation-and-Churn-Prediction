.PHONY: data features segments train evaluate dashboard clean

data:
	python src/data/load_db.py
	python src/data/make_dataset.py

features:
	python src/features.py

segments:
	python -m src.segmentation

train:
	python -m src.model

dashboard:
	python -m src.dashboard_data
	streamlit run app.py

evaluate:
	python src/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
