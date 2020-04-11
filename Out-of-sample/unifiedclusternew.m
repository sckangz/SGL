
function [result,label]=unifiedclusternew(K,A,s,alpha,beta)
% s is the true class label.
[~,n]=size(K);
[~,m]=size(A);
Z=ones(n,m)/m;
c=length(unique(s));
W=zeros(n,m);
options = optimset( 'Algorithm','interior-point-convex','Display','off');
fprintf('iter:%d\n',m);
for i=1:100
    Zold=Z;
    D1=(sum(Z')).^(-1/2);
    D2=(sum(Z)).^(-1/2);
    D=[D1,D2];
    [~,U,V,~,~,~,]=svd2uv(Z, c);
    F=[U;V];
   for ij=1:n
        for ji=1:m
            W(ij,ji)=(norm(F(ij,:)*D(ij)-F(ji,:)*D(n+ji)))^2;
   
        end
   end
   H=2*alpha*eye(m)+2*A'*A;
   H=(H+H')/2;
   parfor ij=1:n
        ff=beta*W(ij,:)-2*(K(:,ij))'*A;
        Z(ij,:)=quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),Z(ij,:),options);
   end
    
    if i>5 &((norm(Z-Zold)/norm(Zold))<1e-3)
        break
    end
    
end
 rng(5489,'twister');
actual_ids= litekmeans(U,c,'MaxIter', 100,'Replicates',100);
label=actual_ids;
[result] = ClusteringMeasure( actual_ids,s);
