function load_images()
    img_path = uigetdir('.', 'Choose Directory of Images');
    names = {'train','test','val'};
    % Start by checking that all the folders exist.
    for i = 1:3
       if (~isfolder([img_path filesep names{i}]))
           error("Invalid Data Directory:%s does not exist,", name);
       end
    end
    
    for i = 1:3
        imgs_normal = dir([img_path filesep names{i}]);
    end
end