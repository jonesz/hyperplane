BINARIES = opt_test

test: opt_test.fut
	futhark test $<

.PHONY: clean

clean:
	$(RM) *.actual *.expected *.c $(BINARIES)
