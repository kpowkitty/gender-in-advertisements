#!/usr/bin/env bash

read -p "Enter the git repository name: " name

if [ -f .gitignore ]; then
    rm -rf .gitignore
fi
if [ -d .git ]; then
    rm -rf .git
fi
if [ -f .gitmodules ]; then
    rm -rf .gitmodules
fi

cat > .gitignore <<EOF
/*
!.gitignore
!pytorch.txt
!test3.py
!test2.py
!test.py
EOF

git init
git add -A
git commit -m "first commit"
git branch -M main
git remote add origin "git@github.com:kpowkitty/$name.git"
git push -u origin main
