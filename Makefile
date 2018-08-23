update-toml: Cargo.toml rc/Cargo.toml
.PHONY: update-toml

build-releases: build-releases.sh update-toml
	./build-releases.sh
.PHONY: build-releases

release: build-releases
	cd dist/latest/im && cargo publish
	cd dist/latest/im-rc && cargo publish
.PHONY: release

Cargo.toml: Cargo.toml.in VERSION Makefile
	m4 -D BASE=. -D "VERSION=$$(< VERSION)" -D CRATE=im -D DESCRIPTION="Immutable collection datatypes" $< > $@

rc/Cargo.toml: Cargo.toml.in VERSION Makefile
	m4 -D BASE=.. -D "VERSION=$$(< VERSION)" -D CRATE=im-rc -D DESCRIPTION="Immutable collection datatypes (the fast but not thread safe version)" $< > $@
