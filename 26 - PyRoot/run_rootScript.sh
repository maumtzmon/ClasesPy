#!/bin/bash

search_dir= /home/oem/Documentos # /the/path/to/base/dir
for entry in "$search_dir"/*;
do
  echo "$entry"
done


for i in 1 2 3 4 5
do
   echo "Welcome $i times"
done
