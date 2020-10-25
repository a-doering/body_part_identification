% Label all files in subdirectories by 
%% Preparing the data from the directories
% Get a list of all files and folders in this folder.
files = dir('/home/s1283/no_backup/s1283/data/3T/')

% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir] & ~strcmp({files.name},'.') & ~strcmp({files.name},'..');

% Extract only those that are directories
subFolders = files(dirFlags)

% Print folder names to command window.
% for k = 1 : length(subFolders)
%   fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
% end

%% Calling the GUI

for idx = 1 : length(subFolders)
    display("idx = " + idx);
    currentPath = [subFolders(idx).folder filesep subFolders(idx).name];
    currentName = 'rework.mat';
    hGUI = CoronalLabelGUI;
    
    % Waits until the window is closed. Window is also closed by saving.
    waitfor(hGUI)
end

% Close the open figure window
close 
display("All files have been opened by the GUI");

%% Testing and getting the names of faulty directories
% Load rework file to check for errors 
% Such as missing info field
testIdx = 2;
testLoad = load([subFolders(testIdx).folder filesep subFolders(testIdx).name filesep 'rework.mat']);
display(testLoad.auxsave.id)
testLoad.info

% Display names of all dirs with missing info field
missingInfoIdxs = [2]; % manually insert idices of faulty files
missingInfoNames = strings(length(missingInfoIdxs),1);
for i = 1 : length(missingInfoIdxs)
    missingInfoNames(i) = subFolders(missingInfoIdxs(i)).name;
end
display(missingInfoNames)