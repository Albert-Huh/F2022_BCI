% Batch exporting of .gdf files to Brain Vision file format
% (.dat/.vhdr/.vmrk) 
% Hyonyoung Shin 10/30/2022 (hyonyoung.shin@utexas.edu) 

eeglab; 

rootdir = 'C:\Users\mcvai\F2022_BCI\ErrP_data';
filelist = dir(fullfile(rootdir, '**\*\*.gdf'));  %get list of files and folders in any subfolder
filelist = filelist(~[filelist.isdir]);  %remove folders from list
bvadir = 'C:\Users\mcvai\F2022_BCI\ErrP_data\bva';

failed_files = cell(1,1);

for i = 1:size(filelist, 1)
    try
        f = strcat(filelist(i).folder, '/', filelist(i).name); 
        gdf = pop_biosig(f);
    
        [status, ~, ~] = mkdir(filelist(i).folder);
    
        if status 
            disp("Successful")
        elseif ~status
            disp("Failed to refer to directory")
        end
    
        name = strcat(filelist(i).folder, '\', filelist(i).name); 
        name = name(1:end-4);
        disp(name)
        bva = pop_writebva(gdf, name);
    catch
        failed_files{1, end+1} = name;
    end
end

