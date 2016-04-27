#!/bin/bash

git stash
git checkout gh-pages

cabal run site rebuild
cd ..:w
cp -r ./blog/_site/* .
git add ./*.html ./css ./images ./posts
git commit -am "Blog update."
git push

git checkout master
git stash apply

