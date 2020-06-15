#!/bin/bash

fileId="1cjEKk15u5jrjUupMm--vTLPdQLlOS8hV"
fileName="pretrained_models.tar.xz"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

tar -xvf ${fileName} -C pretrained_models/