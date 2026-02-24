
% Distance correlation calculation with permutation p-value determination
% for pairwise and multifeature datasets
% M.Cuperlovic-Culf, Canada, February 2026
%% Example usage and testing

% Example 1: Test with linearly related variables
fprintf('=== Example 1: Linear Relationship ===\n');
n = 100;
X1 = randn(n, 1);
Y1 = 2*X1 + randn(n, 1)*0.5;  % Linear with noise

[dcor1, pval1] = distanceCorrelation(X1, Y1, 1000);
pearson1 = corr(X1, Y1);

fprintf('Distance Correlation: %.4f (p = %.4f)\n', dcor1, pval1);
fprintf('Pearson Correlation: %.4f\n', pearson1);

% Example 2: Test with nonlinear relationship
fprintf('\n=== Example 2: Nonlinear Relationship (Quadratic) ===\n');
X2 = randn(n, 1);
Y2 = X2.^2 + randn(n, 1)*0.5;  % Quadratic

[dcor2, pval2] = distanceCorrelation(X2, Y2, 1000);
pearson2 = corr(X2, Y2);

fprintf('Distance Correlation: %.4f (p = %.4f)\n', dcor2, pval2);
fprintf('Pearson Correlation: %.4f (misses nonlinear relationship)\n', pearson2);

% Example 3: Test with independent variables
fprintf('\n=== Example 3: Independent Variables ===\n');
X3 = randn(n, 1);
Y3 = randn(n, 1);  % Independent

[dcor3, pval3] = distanceCorrelation(X3, Y3, 1000);
pearson3 = corr(X3, Y3);

fprintf('Distance Correlation: %.4f (p = %.4f)\n', dcor3, pval3);
fprintf('Pearson Correlation: %.4f\n', pearson3);

% Example 4: Multivariate case
fprintf('\n=== Example 4: Multivariate ===\n');
X4 = randn(n, 3);  % 3 features
Y4 = randn(n, 2);  % 2 features
Y4 = Y4 + 0.5*[X4(:,1), X4(:,2)];  % Some dependence

[dcor4, pval4] = distanceCorrelation(X4, Y4, 1000);

fprintf('Distance Correlation: %.4f (p = %.4f)\n', dcor4, pval4);

%% Visualization
figure('Position', [100, 100, 1400, 400]);

% Plot 1: Linear relationship
subplot(1, 3, 1);
scatter(X1, Y1, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('X', 'FontWeight', 'bold');
ylabel('Y', 'FontWeight', 'bold');
title(sprintf('Linear Relationship\ndCor=%.3f, Pearson=%.3f\np=%.4f', ...
    dcor1, pearson1, pval1), 'FontSize', 11);
grid on;

% Plot 2: Nonlinear relationship
subplot(1, 3, 2);
scatter(X2, Y2, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('X', 'FontWeight', 'bold');
ylabel('Y', 'FontWeight', 'bold');
title(sprintf('Nonlinear Relationship\ndCor=%.3f, Pearson=%.3f\np=%.4f', ...
    dcor2, pearson2, pval2), 'FontSize', 11);
grid on;

% Plot 3: Independent variables
subplot(1, 3, 3);
scatter(X3, Y3, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('X', 'FontWeight', 'bold');
ylabel('Y', 'FontWeight', 'bold');
title(sprintf('Independent Variables\ndCor=%.3f, Pearson=%.3f\np=%.4f', ...
    dcor3, pearson3, pval3), 'FontSize', 11);
grid on;

sgtitle('Distance Correlation Examples', 'FontSize', 14, 'FontWeight', 'bold');

function [dcor, pval] = distanceCorrelation(X, Y, nPermutations)
% Calculate distance correlation and p-value between X and Y
%
% Inputs:
%   X - First variable (n x p matrix, n samples, p features)
%   Y - Second variable (n x q matrix, n samples, q features)
%   nPermutations - Number of permutations for p-value (default: 1000)
%
% Outputs:
%   dcor - Distance correlation coefficient (0 to 1)
%   pval - P-value from permutation test

    if nargin < 3
        nPermutations = 1000;
    end
    
    % Ensure X and Y are matrices
    if isvector(X)
        X = X(:);
    end
    if isvector(Y)
        Y = Y(:);
    end
    
    n = size(X, 1);
    
    if size(Y, 1) ~= n
        error('X and Y must have the same number of samples');
    end
    
    % Calculate observed distance correlation
    dcor = calculateDCor(X, Y);
    
    % Permutation test for p-value
    if nPermutations > 0
        permDcor = zeros(nPermutations, 1);
        
        for perm = 1:nPermutations
            % Randomly permute Y
            permIdx = randperm(n);
            YPerm = Y(permIdx, :);
            
            % Calculate distance correlation for permuted data
            permDcor(perm) = calculateDCor(X, YPerm);
        end
        
        % Calculate p-value (proportion of permutations >= observed)
        pval = (sum(permDcor >= dcor) + 1) / (nPermutations + 1);
    else
        pval = NaN;
    end
end

function dcor = calculateDCor(X, Y)
    % Calculate distance correlation coefficient
    
    n = size(X, 1);
    
    % Calculate pairwise distance matrices
    A = squareform(pdist(X));
    B = squareform(pdist(Y));
    
    % Double centering
    A = doubleCentering(A);
    B = doubleCentering(B);
    
    % Calculate distance covariance
    dCovXY = sqrt(mean(A(:) .* B(:)));
    
    % Calculate distance variances
    dVarX = sqrt(mean(A(:) .^ 2));
    dVarY = sqrt(mean(B(:) .^ 2));
    
    % Calculate distance correlation
    if dVarX > 0 && dVarY > 0
        dcor = dCovXY / sqrt(dVarX * dVarY);
    else
        dcor = 0;
    end
end

function A_centered = doubleCentering(A)
    % Double center a distance matrix
    
    n = size(A, 1);
    
    % Row means
    rowMeans = mean(A, 2);
    
    % Column means
    colMeans = mean(A, 1);
    
    % Grand mean
    grandMean = mean(A(:));
    
    % Double centering
    A_centered = A - rowMeans - colMeans + grandMean;
end






%% Additional function: Distance correlation matrix for multiple variables
function dcorMatrix = distanceCorrelationMatrix(data, nPermutations)
    % Calculate pairwise distance correlations for all columns in data
    %
    % Inputs:
    %   data - n x p matrix (n samples, p variables)
    %   nPermutations - Number of permutations for p-values (default: 1000)
    %
    % Outputs:
    %   dcorMatrix - Structure with fields:
    %       .dcor - p x p matrix of distance correlations
    %       .pval - p x p matrix of p-values
    
    if nargin < 2
        nPermutations = 1000;
    end
    
    p = size(data, 2);
    dcorMatrix.dcor = eye(p);
    dcorMatrix.pval = ones(p);
    
    fprintf('Computing pairwise distance correlations...\n');
    
    for i = 1:p
        for j = (i+1):p
            [dcorMatrix.dcor(i,j), dcorMatrix.pval(i,j)] = ...
                distanceCorrelation(data(:,i), data(:,j), nPermutations);
            
            % Matrix is symmetric
            dcorMatrix.dcor(j,i) = dcorMatrix.dcor(i,j);
            dcorMatrix.pval(j,i) = dcorMatrix.pval(i,j);
        end
        fprintf('  Completed variable %d/%d\n', i, p);
    end
end

%% Example: Distance correlation matrix
fprintf('\n=== Example 5: Distance Correlation Matrix ===\n');
% Create data with various relationships
n = 100;
dataMatrix = zeros(n, 4);
dataMatrix(:,1) = randn(n, 1);                          % Independent
dataMatrix(:,2) = 2*dataMatrix(:,1) + randn(n,1)*0.3;  % Linear with var 1
dataMatrix(:,3) = dataMatrix(:,1).^2 + randn(n,1)*0.3; % Quadratic with var 1
dataMatrix(:,4) = randn(n, 1);                          % Independent

dcorMat = distanceCorrelationMatrix(dataMatrix, 500);

fprintf('\nDistance Correlation Matrix:\n');
disp(dcorMat.dcor);

fprintf('\nP-value Matrix:\n');
disp(dcorMat.pval);

% Visualize correlation matrix
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
imagesc(dcorMat.dcor);
colorbar;
title('Distance Correlation Matrix', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Variable');
ylabel('Variable');
set(gca, 'XTick', 1:4, 'YTick', 1:4);
colormap('hot');
caxis([0 1]);

% Add text values
for i = 1:4
    for j = 1:4
        text(j, i, sprintf('%.3f', dcorMat.dcor(i,j)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
            'Color', dcorMat.dcor(i,j) > 0.5, 'white' : 'black');
    end
end

subplot(1, 2, 2);
imagesc(dcorMat.pval);
colorbar;
title('P-value Matrix', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Variable');
ylabel('Variable');
set(gca, 'XTick', 1:4, 'YTick', 1:4);
colormap('hot');
caxis([0 0.1]);

% Add text values
for i = 1:4
    for j = 1:4
        if dcorMat.pval(i,j) < 0.001
            textStr = '<.001';
        else
            textStr = sprintf('%.3f', dcorMat.pval(i,j));
        end
        text(j, i, textStr, 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'FontSize', 9);
    end
end
