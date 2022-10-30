%% fdgfd

% *Q: why does the performance differ so significantly for the training and
% test set?* 
% In this expriement the training process is simply a linear model
% using Moore-Penrose pseudoinverse. Since the coeficients of linear model are determined 
% by using only training set, it is obvious that fitted linear model can
% significantly predict better the training set rather than test set. Let *X* be the
% training fold and *Y* be the output variable (in this case, torque). 
% The coeficients *W* are obtained by the following equation where X is the  
%% sd 
% and predicted output variable is . This coeficients 
%
% $$W=X^{p} Y  \quad , where \: X^{p} \: is \: the \: Moore-Penrose \: pseudoinverse$$ 
% 
% $$X^{p}$$
%
% Moore-Penrose pseudoinverse
% 


%% 
% the predicted output variable is obtained by this equation
%
% $$\hat{Y}= X W = X X^{p} Y $$
% 