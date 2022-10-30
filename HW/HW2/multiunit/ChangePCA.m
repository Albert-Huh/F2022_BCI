function [Etr,Ete]=ChangePCA(Tr,Te,C)
set_train(Tr);
set_test(Te);
for i=1:length(C)
    set_pca(C(i));
    [Etr(i,:),Ete(i,:)]=train;
end