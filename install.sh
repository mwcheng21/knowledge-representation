#!/bin/bash

function download_from_drive() {
    fileid=${1};
    directory=${2};
    mkdir -p $directory;
    cd $directory;
    echo "Downloading $directory";
    FILE="${directory}.zip";
    if [[ -f "$FILE" ]]; then
        echo "$FILE exists, skipping download"
    else
        curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' \
        ./cookie`&id=${fileid}" -o ${FILE}
        rm ./cookie
    fi
    unzip ${FILE} && rm ${FILE};
    cd ..;
}

function download_from_link() {
    link=${1}
    directory=${2};
    FILE="${3}.zip";

    echo $FILE

    mkdir -p $directory;
    cd $directory;
    echo "Downloading $directory";

    if [[ -f "$FILE" ]]; then
        echo "$FILE exists, skipping download"
    else 
        curl "${link}" --output $FILE;
    fi
    unzip ${FILE} && rm ${FILE};
    cd ..;
}

function download_models() {
    fileid="1_NXItk_1n3IqbiucVYRW85YixPfr21Vq";
    download_from_drive $fileid "models";
}

function download_data() {
    # TODO More Dataset
    addr="https://raw.githubusercontent.com/Leahcim-1/knowledge-representation-dataset/main/medium/krp_medium.zip";
    file="krp_medium"
    download_from_link $addr "data" $file;
}

pip install -r requirements.txt;

download_data;
## download_models;\