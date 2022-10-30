function [Etr,Ete]=ChangeDelay(Tr,Te)
set_test(Te);
set_train(Tr);
for i=1:20
    i
    tic;
    set_delays(1:i) 
    [Etr(i,:),Ete(i,:)]=train;
    toc
    save('q6_1', 'Etr', 'Ete');
end