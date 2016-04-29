#!/bin/bash


cabal run site rebuild
git stash
git checkout gh-pages
echo `pwd`
cp -r ./blog/_site/* .
git add ./*.html ./css ./images ./posts
git commit -am "Blog update."
git push

git checkout master
git stash apply

