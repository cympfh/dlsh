build:
	mkdir -p bin/
	cd dlsh && make build
	for f in `find dlsh/target/release -maxdepth 1 -type f`; do [ -x $$f ] && cp $$f bin/ || true; done
	cp sh/* bin/

clean:
	rm -rf bin
