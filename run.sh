#!/bin/bash

echo 'get ready to be disappointed! running numerical recipes assignment 1.'
echo 'please wait while the computer gets ready............................'
echo "let's download the data files for no reason.........................."
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m11.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m12.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m13.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m14.txt
wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m15.txt

python3 handin1.py > handin1.txt

echo 'alright! handin1.py has ran.'


pdflatex nur.tex
