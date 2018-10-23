%function load_data()
    %% Parameters
    data_path = uigetdir('.', 'Choose Directory of Images');
    x_min = 0.1; x_max = 0.9;
    y_min = 0.2; y_max = 1.0;
    h_out = 224;
    w_out = 224;
    
    %% Initialization
    names = {'val', 'test', 'train'};
    fold_path = cell(1,3);
    norm_path = cell(1,3);
    pneu_path = cell(1,3);
    norm_data = cell(1,3);
    pneu_data = cell(1,3);
    
    %% Start by checking that all the folders exist.
    for i = 1:3
        fold_path{i} = fullfile(data_path, names{i});
        norm_path{i} = fullfile(fold_path{i}, "NORMAL");
        pneu_path{i} = fullfile(fold_path{i}, "PNEUMONIA");
        if (~(isfolder(norm_path{i})))
            error("Invalid Data Directory: %s does not exist.", norm_path{i});
        end
        if (~(isfolder(pneu_path{i})))
            error("Invalid Data Directory: %s does not exist.", pneu_path{i});
        end
    end
    
    %% Load images into .mat files
    for i = 1:3
        files = dir(fullfile(norm_path{i}, "*.jpeg"));
        data = zeros(length(files), h_out*w_out);
        for im_id = 1:length(files)
            file = files(im_id);
            im_origin = imread([file.folder filesep file.name]);
            [H, W] = size(im_origin);
            rect = [round(W*x_min)+1, round(H*y_min), round(W*(x_max-x_min)), round(H*(y_max-y_min))];
            im_crop = imcrop(im_origin, rect);
            im_resize = imresize(im_crop, [h_out, w_out]);
            
%             figure
%             subplot(2,2,1)
%             imshow(im_origin)
%             title('Original Image')
%             subplot(2,2,2)
%             imshow(im_crop)
%             title('Cropped Image')
%             subplot(2,2,3)
%             imshow(im_resize)
%             title('Resized Image')
%             pause

            data(im_id, :) = double(reshape(im_resize, 1, h_out*w_out))/255.0;
        end
        norm_data{i} = data;
        
        files = dir(fullfile(pneu_path{i}, "*.jpeg"));
        data = zeros(length(files), h_out*w_out);
        for im_id = 1:length(files)
            file = files(im_id);
            im_origin = imread([file.folder filesep file.name]);
            [H, W] = size(im_origin);
            rect = [round(W*x_min)+1, round(H*y_min), round(W*(x_max-x_min)), round(H*(y_max-y_min))];
            im_crop = imcrop(im_origin, rect);
            im_resize = imresize(im_crop, [h_out, w_out]);
            data(im_id, :) = double(reshape(im_resize, 1, h_out*w_out))/255.0;
        end
        pneu_data{i} = data;
    end
%end