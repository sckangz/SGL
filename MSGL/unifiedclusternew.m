
function [result]=unifiedclusternew(K,A,s,alpha,beta,gamma,nv)
% s is the true class label.
[~,n]=size(K{1});
[~,m]=size(A{1});
Z=ones(n,m)/m;
c=length(unique(s));
W=zeros(n,m);
for i=1:nv
    av(i)=1/nv;
end
options = optimset( 'Algorithm','interior-point-convex','Display','off');
for i=1:30
    Zold=Z;
    D1=(sum(Z')).^(-1/2);
    D2=(sum(Z)).^(-1/2);
    D=[D1,D2];
    [~,U,V,~,~,~,]=svd2uv(Z, c);
    F=[U;V];
   for ij=1:n
        for ji=1:m
            
            W(ij,ji)=(norm(((F(ij,:)*D(ij)-F(n+ji,:)*D(n+ji))),'fro'))^2;
        end
   end
   
   tmp=av(1)*(A{1})'*(A{1});
   
   for ii=2:nv
        tmp=tmp+av(ii)*(A{ii})'*(A{ii});
   end
   
   H=2*alpha*eye(m)+2*tmp;
   H=(H+H')/2;
   tmp1={};
   for ii=1:n
       
       tmp1{ii}=av(1)*(K{1}(:,ii))'*A{1};
       for tmp2=2:nv
       tmp1{ii}=tmp1{ii}+av(tmp2)*(K{tmp2}(:,ii))'*A{tmp2};
       end
   end
   parfor ij=1:n
       
        
        ff=beta*W(ij,:)-2*tmp1{ij};
        Z(ij,:)=quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),Z(ij,:),options);
   end
   for ii=1:nv
       av(ii)=(-((norm((K{ii}-A{ii}*Z'),'fro')^2))/gamma)^(1/(gamma-1));
   end
    if i>5 &((norm(Z-Zold,'fro')/norm(Zold,'fro'))<1e-3)
        break
    end
end
rng(5489,'twister');
actual_ids= litekmeans(U,c,'MaxIter', 100,'Replicates',100);

[result] = ClusteringMeasure( actual_ids,s);
