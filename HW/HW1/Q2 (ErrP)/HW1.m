clear; close all;
eeglab;  % topoplot function depends on eeglab
load('ErrP_data_HW1.mat');
load('ErrP_channels.mat');

% find nearest indices to -0.2s and 0.8s 
[~, start_idx] = min(abs(params.epochTime+0.2));
[~, end_idx] = min(abs(params.epochTime-0.8));

fs = params.fsamp; 
time = params.epochTime(start_idx:end_idx);
data = trainingEpochs.rotation_data(start_idx:end_idx, :, :);
data_Cz = data(:, params.channelPlot, :);

data_magnitude = cell(1,5); 
Cz_magnitude = cell(1,5);

for i = 1:length(data_magnitude)
    data_magnitude{i} = data(:, :, trainingEpochs.magnitude==(i-1)*3);  % put data into different magnitudes
    Cz_magnitude{i} = data_Cz(:, :, trainingEpochs.magnitude==(i-1)*3);
end
%% 2-3. Grand average Cz 
for i = 1:length(Cz_magnitude)
figure;
hold on
plot(time, mean(Cz_magnitude{i}, 3)); % plot mean across trials dimension = grand average

xline(0, '--')
xlabel('time (s)')
ylabel('amplitude (\muV)')
xlim([-0.2, 0.8]);
title(sprintf('Cz grand average, error = %d^o', (i-1)*3));
hold off
end
%% 2-4. ERN and Pe 
if 1
    for j = 1:5
        disp(j)
        [pks,locs] = findpeaks(mean(Cz_magnitude{j}, 3)); %% this was used to find the peaks semi-manually
        disp(time(locs))  % display times for manual comparison with grand average plot 
        disp(locs')
    end
end

pe = [0.5879, 0.4277, 0.4922, 0.4043, 0.3906];
ne = [0.2793, 0.2852, 0.3105, 0.2676, 0.2793];

for i = 1:5
    figure(i+1)
    hold on 
    xline(pe(i), 'b--')
    xline(ne(i), 'r--')
    text(pe(i)-0.05,-0.5,'Pe'); 
    text(ne(i)+0.015,0.5,'ERN'); 
end
%% 2-5. 
% ne_locs = [246, 249, 262, 240, 246];
% pe_locs = [404, 322, 355, 310, 303]; 

data_error = cell(1, 2); 
Cz_error = cell(1, 2);

for i = 1:length(data_error)
    data_error{i} = data(:, :, trainingEpochs.label==(i-1));  % put data into whether have error or not 
    Cz_error{i} = data_Cz(:, :, trainingEpochs.label==(i-1));
end


for i = 2  % only for error 
figure;
hold on
plot(time, mean(Cz_error{i}, 3)); % plot mean across trials dimension = grand average

xline(0, '--')
xlabel('time (s)')
ylabel('amplitude (\muV)')
xlim([-0.2, 0.8]);
title('Cz grand average, error present');
hold off
end

[pks,locs] = findpeaks(mean(Cz_error{2}, 3)); %% this was used to find the peaks semi-manually
disp(time(locs))  % display times for manual comparison with grand average plot 
disp(locs')

ne_all = 0.2793;
ne_all_loc = 246;
pe_all = 0.4062; 
pe_all_loc = 311; 
 

% trainingEpochs.rotation_data(start_idx:end_idx, :, :);


trial_avg = mean(data_error{2}, 3);  % preserve all channels 
neg_snapshot = trial_avg(ne_all_loc, :);  % data vector at peak time point
pos_snapshot = trial_avg(pe_all_loc, :);

figure; 
topoplot(pos_snapshot, params.chanlocs, 'electrodes', 'labels', 'style' ,'both');
title('At pe, all error trials averaged')
colorbar;

figure; 
topoplot(neg_snapshot, params.chanlocs, 'electrodes', 'labels', 'style' ,'both');
title('At ERN, all error trials averaged')
colorbar;

%% 2-6. 
CAR_data_magnitude = cell(1,5);
CAR_Cz_data_magnitude = cell(1,5);
for i = 1:length(data_magnitude) % 5
    average = mean(data_magnitude{i}, 2);
    for j = 1:size(data_magnitude{i}, 3)
        CARj = data_magnitude{i}(:, :, j) - average(:, 1, j);
        CAR_data_magnitude{i}(:, :, j) = CARj;
    end
    CAR_Cz_data_magnitude{i} = CAR_data_magnitude{i}(:, 15, :);
end



for i = 1:length(CAR_Cz_data_magnitude)
figure;
hold on
plot(time, mean(CAR_Cz_data_magnitude{i}, 3)); % mean across trials dimension = grand average

xline(0, '--')
xlabel('time (s)')
ylabel('amplitude (\muV)')
xlim([-0.2, 0.8]);
title(sprintf('Cz grand average with CAR, error = %d^o', (i-1)*3));
hold off
end

for i = 1:5
    figure(i+9)
    hold on 
    xline(pe(i), 'b--')
    xline(ne(i), 'r--')
    text(pe(i)-0.05,-0.5,'Pe'); 
    text(ne(i)+0.015,0.5,'ERN'); 
end

% To plot required topoplot, we need ERN and Pe timepoints using all error =
% 1 trials 


CAR_data_error = cell(1,2);
CAR_Cz_data_error = cell(1,2);

for i = 1:length(CAR_data_error) % 2  % for each error label
    average = mean(data_error{i}, 2);  % average across channels
    for j = 1:size(data_error{i}, 3)  % repeat for each trial
        CARj = data_error{i}(:, :, j) - average(:, 1, j);
        CAR_data_error{i}(:, :, j) = CARj;
    end
    CAR_Cz_data_error{i} = CAR_data_error{i}(:, 15, :);
end

for i = 2  % only for error trials 
figure;
hold on
plot(time, mean(CAR_Cz_data_error{i}, 3)); % plot mean across trials dimension = grand average

xline(0, '--')
xlabel('time (s)')
ylabel('amplitude (\muV)')
xlim([-0.2, 0.8]);
title('Cz grand average with CAR, error present');
hold off
end

[pks,locs] = findpeaks(-mean(CAR_Cz_data_error{2}, 3)); %% this was used to find the peaks semi-manually
disp(time(locs))  % display times for manual comparison with grand average plot 
disp(locs')

pe_all = 0.4082; 
pe_all_loc = 312; 
ne_all = 0.3008; 
ne_all_loc = 257; 

trial_avg = mean(CAR_data_error{2}, 3);  % preserve all channels 
neg_snapshot = trial_avg(ne_all_loc, :);  % data vector at peak time point
pos_snapshot = trial_avg(pe_all_loc, :);

figure; 
topoplot(pos_snapshot, params.chanlocs, 'electrodes', 'labels', 'style' ,'both');
title('At pe, with CAR, all error trials averaged')
colorbar;

figure; 
topoplot(neg_snapshot, params.chanlocs, 'electrodes', 'labels', 'style' ,'both');
title('At ERN, with CAR, all error trials averaged')
colorbar;

%% 2-7. CCA
X = []; Y = []; 
for class = 1:length(data_error)
    % Craft X 
    dat = data_error{class}; % m samples * n channels * k trials 
    dat = permute(dat, [2, 1, 3]); % n * m * k
    X_class = [];
    for k = 1:size(dat, 3)
        X_class = [X_class, dat(:, :, k)]; 
    end

    % Craft Y 
    grandavg = mean(data_error{class}, 3); 
    grandavg = permute(grandavg, [2, 1]); % n * m

    Y_class = []; 
    for k = 1:size(dat, 3) % replicate k times 
        Y_class = [Y_class, grandavg]; 
    end

    X = [X, X_class];
    Y = [Y, Y_class];
end

[A,B,r,U,V] = canoncorr(X', Y'); % computes the sample canonical coefficients for the data matrices X and Y.
% A is W_x in the paper 

% First 5 filters 
% spat = A(1:5, :); 
% figure;
% for i = 1:size(spat, 1)
%     subplot(1,5,i)
%     W = spat(i, :);
%     topoplot(W, params.chanlocs, 'electrodes', 'labels', 'style' ,'both');
%     title(sprintf('Spatial filter %d', i));
% end
% colorbar;

%%
spat = A(:, 1:5); % row is channel, the weights are contained in each column (refer to canoncorr live documentation)
figure;

for i = 1:size(spat, 2)
    subplot(1,5,i)
    W = spat(:, i);
    topoplot(W, params.chanlocs, 'electrodes', 'on', 'style' ,'both');
    title(sprintf('Spatial filter %d', i));
end
cb = colorbar('east');
cb.Position(1) = cb.Position(1) + 0.09;