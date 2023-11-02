function [X] = DL(G,b_analytic,Nodes)
sparsity = 4;
numnodes = size(Nodes,1);

D = G;
for ii = 1:size(G,2)
    D(:,ii)=D(:,ii)/norm(D(:,ii));
end

% X = pinv(G)*b_analytic;
K_max = 100;
lambda = 0.01;
Y = b_analytic;
X0 = zeros(numnodes, 1);

%分组
% for i = 1:NumNode-1
%     [Idx,~] = knnsearch(Nodes, Nodes(i,:),'K',4);
%     Gro(i,Idx) = 1;
% end

for i = 1:K_max
    % 解决字典学习中的稀疏系数β
    err0 = norm(G*X0-b_analytic);
    X = OMP(D,Y,sparsity);
    % 解决字典学习中的字典D
    D = KSVD(Y, D, X);
    % 获得重建源X
%     X = (G'*G + lambda)\(A'*b_analytic + lambda * D * beta);
    err1 = norm(G*X-b_analytic);
    if abs(err1-err0)<1e-5
        X_idx = find(X);
        X = adjust_output(Nodes, X_idx);
        break;
    end
    X0 = X;
end
end

function X = OMP(D,Y,sparsity)
%Step 1
index = []; 
k = 1; 
[Dm, Dn] = size(D); 
r = Y;
X=zeros(Dn,1);
cor = D'*r;
while k <= sparsity
    %Step 2
    [Rm,ind] = max(abs(cor)); 
    index = [index,ind]; 
    %Step 3
    P = D(:,index)*inv(D(:,index)'*D(:,index))*D(:,index)';
    r = (eye(Dm)-P)*Y; 
    cor=D'*r;
    k=k+1;
end
%Step 5
X_ind = inv(D(:,index)'*D(:,index))*D(:,index)'*Y;
X(index) = X_ind;
end

function D = KSVD(Y, D, X)
% 获取系数向量X中不为0的行号
nonzero_index = find(X);
n_comp = size(nonzero_index,1);
for i = 1:n_comp
    if ~isempty(nonzero_index)
        MX = X;
        MX(nonzero_index(i))=0;
        E_i = Y - D*MX;
        [u,s,v] = svd(E_i);
        D(:,nonzero_index(i)) = u(:, 1);
        X(nonzero_index(i)) = s(1,1)*v(1,:);
    end
end
end

function[output] = adjust_output(Nodes, X_idx) 

%找到大于0的中心点
x_center = mean(Nodes(X_idx,1));
y_center = mean(Nodes(X_idx,2));
z_center = mean(Nodes(X_idx,3));

NumNode = size(Nodes,1);
output = zeros(NumNode,1);

% 找到中心点所在组的四个节点
Idx_light = knnsearch(Nodes, [x_center,y_center,z_center],'K',4);
output(Idx_light) = 1;
end