build:
	mkdir -p bin/
	cd dlsh && make build
	for f in `find dlsh/target/release -maxdepth 1 -type f`; do [ -x $$f ] && mv $$f bin/ || true; done
	cp sh/* bin/

clean:
	rm -rf bin
