function [selectedFeatures] = MIM_nfi(X_data,Y_labels, topK, T)
% Summary 
%    MIM algorithm for feature selection
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_labels: n x 1 vector with the labels
%    topK: Number of features to be selected
%    T: datatype
numFeatures = size(X_data,2);
score_per_feature = zeros(1,numFeatures,'like', T.score_per_feature);
for index_feature = 1:numFeatures
    score_per_feature(index_feature) = mi_nfi(X_data(:,index_feature),Y_labels, T);
end

[~,selectedFeatures] = sort(double(score_per_feature),'descend');

selectedFeatures = selectedFeatures(1:topK);