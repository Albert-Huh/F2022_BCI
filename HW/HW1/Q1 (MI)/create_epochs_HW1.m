% clear; close all; clc

function [epoch, epoch_mu_power] = create_epochs_HW1(filename)
[s,h] = sload(filename);
ns = 32;
raw_eeg = s(:,1:ns);
fs = h.SampleRate;
t = ((1:length(raw_eeg))-1)/fs;
%% 1-2: Temporal and spatial filtering

% bpf butter (fir) design
n = 4; % order
mu_fc = [8, 12]; % mu bandwidth
wn = mu_fc/(fs/2);
ftype = 'bandpass';
[b,a] = butter(n,wn,ftype);

raw_mu = filtfilt(b, a, raw_eeg); % zero-phase filtering

% apply car
raw_mu_car = raw_mu - mean(raw_mu,2);

% load montage
load selectedChannels

% get XYZ coordinates in convenient variables
X = [selectedChannels.X];
Y = [selectedChannels.Y];
Z = [selectedChannels.Z];

raw_mu_surf_lap = laplacian_perrinX(raw_eeg',X,Y,Z)';
% surface Laplacian can be computed by the following code at t_ind
% raw_mu_surf_lap = zeros(length(tind),ns);
% for i=1:length(tind)
%     temp = squeeze(raw_mu(i,:)');
%     raw_mu_surf_lap(i,:) = laplacian_perrinX(temp,X,Y,Z)';
% end

%% 1-3 Epoching

event = [h.EVENT.TYP, h.EVENT.POS];
ntrial = sum(h.EVENT.TYP == 1000);
epoch = cell(ntrial,5);
trial_pos = zeros(ntrial,3);
hit = zeros(ntrial,1);
class = zeros(ntrial,1);
move = zeros(ntrial,1);

j = 1;
for i = 1:length(event)
     if h.EVENT.TYP(i) == 1000
          start_pos = h.EVENT.POS(i);
          trial_pos(j,1) = start_pos;
     elseif h.EVENT.TYP(i) == 769 || h.EVENT.TYP(i) == 770
          cue_pos = h.EVENT.POS(i);
          trial_pos(j,2) = cue_pos;
%           if h.EVENT.TYP(i) == 769
%                cue = 769;
%           elseif h.EVENT.TYP(i) == 770
%                cue = 770;
%           end

          cue = h.EVENT.TYP(i);
     elseif h.EVENT.TYP(i) == 7691 || h.EVENT.TYP(i) == 7701
          if (h.EVENT.TYP(i) == 7691) && (cue == 769)
%                disp('LH')
               move(j) = cue;
          elseif (h.EVENT.TYP(i) == 7701) && (cue == 770)
%                disp('RH')
               move(j) = cue;
          else
               move(j) = 0;
          end
     elseif h.EVENT.TYP(i) == 7692 || h.EVENT.TYP(i) == 7702 || h.EVENT.TYP(i) == 7693 || h.EVENT.TYP(i) == 7703
          end_pos = h.EVENT.POS(i);
          trial_pos(j,3) = end_pos;
          class(j) = cue;
          if (h.EVENT.TYP(i) == 7693) && (cue == 769)
%                disp('LH')
               hit(j) = cue;
          elseif (h.EVENT.TYP(i) == 7703) && (cue == 770)
%                disp('RH')
               hit(j) = cue;
          else
               hit(j) = 0;
          end

          j = j+1;
     end
end
for i = 1:ntrial
     epoch_time = (-(trial_pos(i,2)-trial_pos(i,1)):trial_pos(i,3)-trial_pos(i,2))/fs;
     epoch{i,1} = epoch_time';
     epoch{i,2} = raw_mu(trial_pos(i,1):trial_pos(i,3),:);
     epoch{i,3} = raw_mu_car(trial_pos(i,1):trial_pos(i,3),:);
     epoch{i,4} = raw_mu_surf_lap(trial_pos(i,1):trial_pos(i,3),:);
     epoch{i,5} = hit(i);
     epoch{i,6} = class(i);
     epoch{i,7} = move(i);
end

%% 1-4 Compute Mu Power (s.^2)

epoch_mu_power = cell(ntrial,6);
for i = 1:ntrial
     epoch_mu_power{i,1} = epoch{i,1};
     epoch_mu_power{i,5} = epoch{i,5};
     epoch_mu_power{i,6} = epoch{i,6};
     epoch_mu_power{i,7} = epoch{i,7};
     for j = 2:4
          s_epoch = epoch{i,j};
          power = zeros(size(s_epoch));
          for k = 1:ns
               power(:,k) = s_epoch(:,k).^2;
          end
          epoch_mu_power{i,j} = power;
     end
end

end