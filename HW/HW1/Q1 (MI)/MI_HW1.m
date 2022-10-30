clear; close all; clc
addpath('Subject_003_Session_001_TESS_Online_Visual');
nrun = 4;
file_list = [];
for i = 1:nrun
     file = uigetfile('*.gdf'); % match this argument to file type
     if isequal(file,0)
         disp('No file selected');
     else
         disp(['File: ', file]);
     end
     file_list = [file_list; file];
end

%% Concatenate Epoches
all_epoch = [];
all_epoch_mu_power = [];
for i = 1:nrun
     [epoch, epoch_mu_power] = create_epochs_HW1(file_list(i,:));  
     all_epoch = [all_epoch; epoch];
     all_epoch_mu_power = [all_epoch_mu_power; epoch_mu_power];
end
%% 1-5 Topo Plot
fs = 512;
ntrial = length(all_epoch_mu_power);
ns = 32;
n = fs*0.5;

epoch_end_500ms = zeros(n,ns,ntrial);
epoch_car_end_500ms = zeros(n,ns,ntrial);
epoch_lap_end_500ms = zeros(n,ns,ntrial);
power_end_500ms = zeros(n,ns,ntrial);
power_car_end_500ms = zeros(n,ns,ntrial);
power_lap_end_500ms = zeros(n,ns,ntrial);
type = zeros(ntrial,1);
LH_epoch = [];
RH_epoch = [];
j = 1;
k = 1;
for i = 1:ntrial
     epoch_end_500ms(:,:,i) = all_epoch{i,2}(end-n+1:end,:);
     epoch_car_end_500ms(:,:,i) = all_epoch{i,3}(end-n+1:end,:);
     epoch_lap_end_500ms(:,:,i) = all_epoch{i,4}(end-n+1:end,:);
     power_end_500ms(:,:,i) = all_epoch_mu_power{i,2}(end-n+1:end,:);
     power_car_end_500ms(:,:,i) = all_epoch_mu_power{i,3}(end-n+1:end,:);
     power_lap_end_500ms(:,:,i) = all_epoch_mu_power{i,4}(end-n+1:end,:);
     type(i) = all_epoch{i,6};
     if all_epoch{i,7} == 769
          LH_epoch(:,:,j) = all_epoch{i,2}(end-n+1:end,:);
          LH_epoch_car(:,:,j) = all_epoch{i,3}(end-n+1:end,:);
          LH_epoch_lap(:,:,j) = all_epoch{i,4}(end-n+1:end,:);
          j = j+1;
     elseif all_epoch{i,7} == 770
          RH_epoch(:,:,k) = all_epoch{i,2}(end-n+1:end,:);
          RH_epoch_car(:,:,k) = all_epoch{i,3}(end-n+1:end,:);
          RH_epoch_lap(:,:,k) = all_epoch{i,4}(end-n+1:end,:);
          k = k+1;
     end
end

% disp(unique(type))
% epoch_grand_avg = {epoch_end_500ms, epoch_car_end_500ms, epoch_lap_end_500ms, all_epoch{:,5}};
% power_grand_avg = {power_end_500ms, power_car_end_500ms, power_lap_end_500ms, all_epoch{:,5}};
LH_grand_avg = {LH_epoch, LH_epoch_car, LH_epoch_lap};
RH_grand_avg = {RH_epoch, RH_epoch_car, RH_epoch_lap};
for i = 1:3
     LH_grand_avg{1,i} =  mean(LH_grand_avg{1,i},3);
     RH_grand_avg{1,i} =  mean(RH_grand_avg{1,i},3);
end

mu_bandpower = zeros(ns,3);
for i = 1:3
     LH_power = zeros(size(LH_grand_avg{1,i}));
     RH_power = zeros(size(RH_grand_avg{1,i}));
     for j = 1:ns
               LH_power(:,j) = LH_grand_avg{1,i}(:,j).^2;
               RH_power(:,j) = RH_grand_avg{1,i}(:,j).^2;
     end
     LH_mu_bandpower(:,i) = sum(LH_power,1)'/n;
     RH_mu_bandpower(:,i) = sum(RH_power,1)'/n;
end
%%
close all;
load selectedChannels.mat


titles = {sprintf('Voltage, %d ms', 500), sprintf('CAR, %d ms', 500), sprintf('Laplacian, %d ms', 500)};



for i = 1:3
figure;
subplot(1,2,1)
topoplot(LH_mu_bandpower(:,i),selectedChannels, 'electrodes', 'on', 'style' ,'both', 'maplimits', 'maxmin');
title(strcat(titles{i}, ' LH'))

subplot(1,2,2)
topoplot(RH_mu_bandpower(:,i),selectedChannels, 'electrodes', 'on', 'style' ,'both', 'maplimits', 'maxmin');
title(strcat(titles{i}, ' RH'))
end

% subplot(1,3,2)
% figure(2);
% subplot(1,2,1)
% topoplot(LH_mu_bandpower(:,2),selectedChannels, 'electrodes', 'on', 'style' ,'both');
% title(sprintf('CAR, %d ms', 500))
% subplot(1,2,2)
% topoplot(RH_mu_bandpower(:,2),selectedChannels, 'electrodes', 'on', 'style' ,'both');
% title(sprintf('CAR, %d ms', 500))
% 
% % plot Laplacian map (spatially filtered)
% % subplot(1,3,3)
% figure(3);
% subplot(1,2,1)
% topoplot(LH_mu_bandpower(:,3),selectedChannels, 'electrodes', 'on', 'style' ,'both');
% title(sprintf('Laplacian, %d ms', 500))
% subplot(1,2,2)
% topoplot(RH_mu_bandpower(:,3),selectedChannels, 'electrodes', 'on', 'style' ,'both');
% title(sprintf('Laplacian, %d ms', 500))
%% Simulated data
% load selectedChannels
% 
% % get XYZ coordinates in convenient variables
% X = [selectedChannels.X];
% Y = [selectedChannels.Y];
% Z = [selectedChannels.Z];
% 
% % tind = 1:1000:length(t);
% % raw_mu_surf_lap = zeros(length(tind),ns);
% % for i=1:length(tind)
% %     temp = squeeze(raw_mu(i,:)');
% %     raw_mu_surf_lap(i,:) = laplacian_perrinX(temp,X,Y,Z)';
% % end
% 
% raw_mu_surf_lap = laplacian_perrinX(raw_eeg',X,Y,Z)';

%% 
% ind = trial_pos(18,3) - fs/2;
% epoch{18,2}(end-fs/2,:)
% figure
% topoplot(squeeze(epoch{18,2}(end-fs/2,:)'),selectedChannels,'plotrad',.53,'maplimits',[-10 10]);
% title([ 'Voltage, ' 500 ' ms' ])
% 
% figure
% topoplot(squeeze(epoch{18,2}(end-fs/2,:)'),selectedChannels,'plotrad',.53,'maplimits',[-10 10]);
% title([ 'CAR, ' 500 ' ms' ])
% 
% % plot Laplacian map (spatially filtered)
% figure
% topoplot(laplacian_perrinX(squeeze(epoch{18,2}(end-fs/2,:)'),X,Y,Z),selectedChannels,'plotrad',.53,'maplimits',[-40 40]);
% title([ 'Lap., ' 500 ' ms' ])