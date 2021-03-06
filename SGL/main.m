clear;


load('tr45_690n_8261d_10c_tfidf_uni.mat');
x=X';
Y=y;
ns=length(unique(Y));

numanchor=[25 30 33 36 40 50];
alpha=[.1 1 10 50];
beta=[1e-4 1e-3];% 1e-2 .1 1 10 100];

for j=1:length(numanchor)
%     rand('twister',5489);
    rng(5489,'twister');
    
    [~, H] = litekmeans(X,numanchor(j),'MaxIter', 100,'Replicates',10);
    
    
    for i=1:length(alpha)
        for m=1:length(beta)
            fprintf('params:\tnumanchor=%d\t\talpha=%f\tbeta:%d\n',numanchor(j), alpha(i), beta(m));
            tic;
            [result]=unifiedclusternew(x,H',Y,alpha(i),beta(m));
            t=toc;
            
            fprintf('result:\t%12.6f %12.6f %12.6f %12.6f\n',[result t]);
            dlmwrite('tr45_690n_8261d_10c_tfidf_uni.txt',[numanchor(j) alpha(i) beta(m) result t],'-append','delimiter','\t','newline','pc');
            
            
        end
    end
end

