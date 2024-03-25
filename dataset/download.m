% Vicente Ordonez @ 2011
% Stony Brook University, State University of New York.
%
% Publication:
% Im2Text: Describing Images Using 1 Million Captioned Photographs.
% NIPS 2011. V. Ordonez, G. Kulkarni, TL. Berg
%
% This file downloads the images in the SBU_captioned_photo_dataset_urls.txt file.

output_directory = 'sbu_images';
urls = textread('SBU_captioned_photo_dataset_urls.txt', '%s', -1);
if ~exist(output_directory, 'dir')
	mkdir(output_directory);
end

rand('twister', 123);
urls = urls(randperm(length(urls)));
for i = 1 : length(urls)
	if ~exist(fullfile(output_directory, [regexprep(urls{i}(24, end), '/', '_')]))
		cmd = ['wget -t 3 -T 5 --quiet ' urls{i} ...
			   ' -O ' output_directory '/' regexprep(urls{i}(24, end), '/', '_')];
		unix(cmd);
		fprintf('%d. %s\n', i, urls{i});
	end	
end

