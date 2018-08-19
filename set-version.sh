#!/usr/bin/env bash

PROJECT_ROOT=`git rev-parse --show-toplevel`
PREVIOUS_CWD=`pwd`

VERSION="$1"
cd $PROJECT_ROOT

# bump versions together
perl -pi -e "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml rc/Cargo.toml

# reset env
cd $PREVIOUS_CWD
