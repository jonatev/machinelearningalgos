function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Set these values correctly
X_norm = X;
mu=mean(X);
sigma=std(X);

X_mu=eye(size(X,1),size(X,2));
X_sigma=eye(size(X,1),size(X,2));
for i=1:size(X,2)
    X_mu(:,i)=mu(i)*ones(size(X,1),1);
    X_sigma(:,i)=(1/sigma(i))*ones(size(X,1),1);
    X_norm(:,i)=(X(:,i)-X_mu(:,i)).*X_sigma(:,i);
end


% ============================================================

end
