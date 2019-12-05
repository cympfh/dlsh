build:
	mkdir -p bin/
	cd dlsh && make build
	for f in `find dlsh/target/release -type f -maxdepth 1`; do [ -x $$f ] && mv $$f bin/ || true; done
