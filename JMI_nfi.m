function [selectedFeatures] = JMI_nfi(X_data,Y_labels, topK, T)
% Summary 
%    JMI algorithm for feature selection
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_labels: n x 1 vector with the labels
%    topK: Number of features to be selected
%    T: datatypes

numFeatures = size(X_data,2);

score_per_feature = zeros(1,numFeatures,'like', T.score_per_feature);
for index_feature = 1:numFeatures
    score_per_feature(index_feature) = mi_nfi(X_data(:,index_feature),Y_labels, T);
end
[~,selectedFeatures(1)]= max(double(score_per_feature));

not_selected_features = setdiff(1:numFeatures,selectedFeatures);

%%% Efficient implementation of the second step, at this point I will store
%%% the score of each feature. Whenever I select a feature I put NaN score
score_per_feature = zeros(1,numFeatures,'like', T.score_per_feature);
score_per_feature(selectedFeatures(1)) = double(-1); %NaN
%NaN cannot be converted to fixed-point. 

count = 2;
while count<=topK

    for index_feature_ns = 1:length(not_selected_features)

            score_per_feature(not_selected_features(index_feature_ns)) = double(score_per_feature(not_selected_features(index_feature_ns)))+double(mi_nfi([X_data(:,not_selected_features(index_feature_ns)),X_data(:, selectedFeatures(count-1))], Y_labels,T));
      
    end
    [~,selectedFeatures(count)]= max(score_per_feature);
    
   score_per_feature(selectedFeatures(count)) = double(-1);
    not_selected_features = setdiff(1:numFeatures,selectedFeatures);
    count = count+1;
end