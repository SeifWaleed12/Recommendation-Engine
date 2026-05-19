demo:
	curl -X POST "http://localhost:8000/api/v1/demo/run" -H "Content-Type: application/json" -d "{\"user_type\":\"cold_user\"}"
	curl -X POST "http://localhost:8000/api/v1/demo/run" -H "Content-Type: application/json" -d "{\"user_type\":\"warm_user\"}"

benchmark:
	python evaluation/benchmark.py

store:
	cd frontend && npm run dev
