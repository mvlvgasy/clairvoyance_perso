install:
	@pip install -e .

run_api:
	uvicorn clairvoyance.api.fast:app --reload

run_streamlit:
	streamlit run frontend/app.py

test:
	@pytest tests

run_train:
	python -c "from clairvoyance.interface.main import train; train()"
