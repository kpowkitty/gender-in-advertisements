rm -rf .gitignore
cat > .gitignore <<EOF
/*
!.gitignore
!pytorch.txt
!test3.py
!test2.py
!test.py
EOF
git rm -r --cached .
git add -A
git commit -m "removing old files"
git push -f origin main
