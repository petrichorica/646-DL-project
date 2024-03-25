SBU Captioned Photo Dataset.
This dataset correspond to the following paper:

Vicente Ordonez, Girish Kulkarni, Tamara L. Berg.
Im2Text: Describing Images Using 1 Million Captioned Photographs.
Neural Information Processing Systems(NIPS), 2011.

FILES:

SBU_captioned_photo_dataset_urls.txt
	This file contains 1 million urls corresponding to each image
    in this dataset. The images point to public images on Flickr.
	Note: Images might be removed by users at anytime.

SBU_captioned_photo_dataset_captions.txt
	This file contains 1 million captions corresponding to each
	image in this dataset. The captions are in the same order 
    as the urls in the above file so for instance caption in line 101
    in this file corresponds to the picture pointed by url in line 101
	in the file above. 

download.m
    This file tries to download the images from the urls specified
    in SBU_captioned_photo_dataset_urls.txt

Vicente Ordonez (vordonezroma@cs.stonybrook.edu)
