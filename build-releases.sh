#!/usr/bin/env bash

PROJECT_ROOT=`git rev-parse --show-toplevel`
PREVIOUS_CWD=`pwd`

VERSION="$1"

# If no version is given use the same
if [ "x$VERSION" == x ]; then
  VERSION=$(perl -ne 'if (/^version = \"(.*)\"/) { print $1 . "\n" }' Cargo.toml)
fi

cd $PROJECT_ROOT

# create a release folder
rm -rf "dist/$VERSION"
mkdir -p "dist/$VERSION"
cd "dist/$VERSION"

# create variations
for VARIATION in im im-rc; do
  mkdir $VARIATION
  cp -a ../../src ../../build.rs ../../README.md $VARIATION
  if [ "$VARIATION" == "im-rc" ]; then
    cp -a ../../rc/Cargo.toml $VARIATION
    find "$VARIATION" -name '*.rs' -exec perl -pi -e "s/\bextern crate im\b/extern crate im_rc as im/g" "{}" \;
  else
    cp -a ../../Cargo.toml $VARIATION
  fi
  perl -pi -e "s/^version = \".*\"/version = \"$VERSION\"/" $VARIATION/Cargo.toml
  perl -pi -e "s/\"..\//\"/g" $VARIATION/Cargo.toml
done

# ln the latest
(cd .. && ln -fs "$VERSION" latest)

# reset env
cd $PREVIOUS_CWD
