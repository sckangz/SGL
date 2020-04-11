

landmarkno=1000;
load('covtype.mat');
load('covtype_landmarks.mat')
x=fea';
Y=gnd(landmark_ID);
datatrain=x(:,landmark_ID(1:landmarkno));
datatest=x(:,landmark_ID(1+landmarkno:end));
ns=length(unique(Y));
numanchor=[ 21 23 24 26 28 30];
alpha=[1];
beta=[  1e-4  ];
for j=1:length(numanchor)
%     rand('twister',5489);
    
    rng(5489,'twister');
    [~, H] = litekmeans(datatrain',numanchor(j),'MaxIter', 100,'Replicates',10);
    
    
    for i=1:length(alpha)
        for m=1:length(beta)
            fprintf('params:\tnumanchor=%d\t\talpha=%f\tbeta:%d\n',numanchor(j), alpha(i), beta(m));
           
            [~,trainlabel]=unifiedclusternew(datatrain,H',Y(1:landmarkno),alpha(i),beta(m));
     
           NNlabel=knnclassify(H,datatrain',trainlabel, 1,  'euclidean', 'nearest' );
           tic;
           %testlabel=knnclassify(datatest',datatrain',trainlabel, 3,  'euclidean', 'nearest' );
           testlabel=knnclassify(datatest',H,NNlabel, 3,  'euclidean', 'nearest' );
        
            trainlabel = reshape(trainlabel,1,[]);
            testlabel = reshape(testlabel,1,[]);
            P_label = [trainlabel testlabel]';
            P_label = reshape(P_label,[],1);
            
            result=ClusteringMeasure(Y,P_label);
            t=toc;
            fprintf('result:\t%12.6f %12.6f %12.6f %12.6f\n',[result t]);
            dlmwrite('covtype.txt',[numanchor(j) alpha(i) beta(m) result t],'-append','delimiter','\t','newline','pc');
            
            
        end
    end
end
