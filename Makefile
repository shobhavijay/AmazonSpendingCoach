.PHONY: install install-forecast-prophet install-forecast-arima install-dev run test clean

install:
	pip install -e .

install-forecast-prophet:
	pip install -e ".[forecast-prophet]"

install-forecast-arima:
	pip install -e ".[forecast-arima]"

install-dev:
	pip install -e ".[dev]"

run:
	python main.py

# Override CSV paths without editing .env:
# make run TRANSACTIONS_CSV=/path/to/orders.csv RETURNS_CSV=/path/to/refunds.csv
run-custom:
	TRANSACTIONS_CSV=$(TRANSACTIONS_CSV) \
	CATEGORIZED_CSV=$(CATEGORIZED_CSV) \
	RETURNS_CSV=$(RETURNS_CSV) \
	CART_CSV=$(CART_CSV) \
	python main.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info/
