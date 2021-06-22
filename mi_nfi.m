function MI= mi_nfi(X, Y, T) %#codegen
% Summary
%    Estimate mutual information I(X;Y) between two categorical variables X,Y
%    X,Y can be matrices which are converted into a joint variable before computation
[~,~,X] = unique(X,'rows');
[~,~,Y] = unique(Y,'rows');
[~,ar,X]=unique(X); arity_X=length(ar);
[~,ar,Y]=unique(Y); arity_Y=length(ar);
n = length(Y);
table_dim = [arity_X arity_Y ];

p_XY = zeros(table_dim,'like', T.p_XY);
p_XY(:) = accumarray([X Y],1,table_dim)/n;

p_X_p_Y=cast(sum(double(p_XY),2) * sum(double(p_XY),1),'like', T.p_X_p_Y);

id_non_zero = intersect(find(double(p_XY)~=0),find(double(p_X_p_Y)~=0));

MI = cast(0.0,'like', T.MI);
MI(:) = sum(sum( double(p_XY(id_non_zero)) .* log(double(p_XY(id_non_zero)) ./  double(p_X_p_Y(id_non_zero)))));