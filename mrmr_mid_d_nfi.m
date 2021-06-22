function [fea] = mrmr_mid_d_nfi(d, f, K, T)
% MID scheme according to MRMR
%
% By Hanchuan Peng
% April 16, 2003
%

bdisp=0;

nd = size(d,2);
nc = size(d,1);

t = zeros(1,nd,'like', T.score_per_feature);
for i=1:nd, 
   t(i) = mi_nfi(d(:,i), f, T);
end; 


[~, idxs] = sort(-double(t));

fea(1) = idxs(1);

KMAX = min(500,nd);

idxleft = idxs(2:KMAX);

k=1;
if bdisp==1,
fprintf('k=1 cost_time=(N/A) cur_fea=%d #left_cand=%d\n', ...
      fea(k), length(idxleft));
end;

mi_array = zeros(nd,K,'like', T.score_per_feature);
t_mi = zeros(1,K-1,'like', T.score_per_feature);
c_mi = zeros(1,K-1,'like', T.score_per_feature);
for k=2:K,
   ncand = length(idxleft);
   curlastfea = length(fea);
   for i=1:ncand,
      t_mi(i) = mi_nfi(d(:,idxleft(i)), f, T); 
      mi_array(idxleft(i),curlastfea) = getmultimi(d(:,fea(curlastfea)), d(:,idxleft(i)), T);
      c_mi(i) = mean(double(mi_array(idxleft(i), 1:(k-1))));
   end;

   [~, fea(k)] = max(double(t_mi(1:ncand)) - double(c_mi(1:ncand)));
   
   tmpidx = fea(k); fea(k) = idxleft(tmpidx); idxleft(tmpidx) = [];

   if bdisp==1,
   fprintf('k=%d cur_fea=%d #left_cand=%d\n', ...
      k, fea(k), length(idxleft));
   end;
end;

return;

%===================================== 
function c = getmultimi(da, dt, T) 
c = zeros(1,size(da,2),'like', T.score_per_feature);
for i=1:size(da,2), 
   c(i) = mi_nfi(da(:,i), dt, T);
end;