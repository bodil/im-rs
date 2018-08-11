#!/usr/bin/env bash

PROJECT_ROOT=`git rev-parse --show-toplevel`
PREVIOUS_CWD=`pwd`

cd $PROJECT_ROOT

# get version from im
VERSION=`sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml`

# generate im-sync
rm -rf im-rc
mkdir im-rc
cp -a src Cargo.toml build.rs README.md im-rc/
sed -ie 's/^name = "im"/name = "im-rc"/' im-rc/Cargo.toml
sed -ie 's/^description = "\(.*\)"/description = "\1 (the fast but not thread safe version)"/' im-rc/Cargo.toml
sed -ie "s/^version = \".*\"/version = \"$VERSION\"/" im-rc/Cargo.toml

# reset env
cd $PREVIOUS_CWD
