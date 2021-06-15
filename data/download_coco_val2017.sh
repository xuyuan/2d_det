#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ ! -z "$1" ]
  then
    # check if specified dir is valid
    if [ ! -d $1 ]; then
        echo $1 " is not a valid directory"
        exit 0
    fi
    echo "navigating to " $1 " ..."
    cd $1
fi

mkdir -p ./coco
cd ./coco

#echo "Downloading MSCOCO train images ..."
#curl -LO http://images.cocodataset.org/zips/train2017.zip
echo "Downloading MSCOCO val images ..."
curl -LO http://images.cocodataset.org/zips/val2017.zip

echo "Downloading MSCOCO train/val annotations ..."
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting val images ..."
unzip val2017.zip
echo "Extracting annotations ..."
unzip annotations_trainval2017.zip

echo "Removing zip files ..."
rm val2017.zip
rm annotations_trainval2017.zip

echo "struct dataset"
cd val2017
mkdir images
mv *.jpg images
mv ../annotations/instances_val2017.json annotations.json
rm -r ../annotations

end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"