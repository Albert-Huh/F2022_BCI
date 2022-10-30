function out = compute_fvaf(patterns_desired, patterns_actual)

% Return the fraction of variance accounted for in <desired> by <actual>

% Squared error
numerator = sum((patterns_desired - patterns_actual).^2);

% Mean desired
mn = mean(patterns_desired);

% Variance of the desired
divisor = sum((patterns_desired - ones(size(patterns_desired,1), 1) * mn).^2);

% Combined measure
out = 1 - numerator ./ divisor;

