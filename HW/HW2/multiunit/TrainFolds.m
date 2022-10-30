function [Etr,Ete]=TrainFolds
set_test(10);
for i=1:7
    i
    set_train(1:i);
    tic;
    [Etr(i,:),Ete(i,:)]=train;
    toc
    save('q5', 'Etr', 'Ete');
end
    